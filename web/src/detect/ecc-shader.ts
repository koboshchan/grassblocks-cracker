/** Component-only OpenCV-style homography ECC. All image and solve work is GPU. */
export const eccShader=String.raw`
struct Params { width:u32,height:u32,count:u32,pad:u32 }
struct State { base:array<f32,9>, warp:array<f32,9>, rho:f32,last:f32,status:f32,iterations:f32,rotation:f32,pad:f32,firstMask:f32,lastMask:f32,minMask:f32,maxMask:f32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var image:texture_2d<f32>;
@group(0) @binding(2) var<storage,read> refs:array<vec4<f32>,256>;
@group(0) @binding(3) var<storage,read_write> state:array<State>;
// Patch planes raw,smoothed,referenceValue,gradientX,gradientY.
@group(0) @binding(4) var<storage,read_write> patches:array<f32>;
// Per-pixel warped input,referenceValue,gx,gy and mask; Jacobian eight floats.
@group(0) @binding(5) var<storage,read_write> pixels:array<f32>;
@group(0) @binding(6) var<storage,read_write> jac:array<f32>;
// 6 statistics then 64 H,8 imageProjection,8 templateProjection.
@group(0) @binding(7) var<storage,read_write> sums:array<f32>;
var<workgroup> reduce:array<f32,128>;
fn finite(v:f32)->bool{return abs(v)<=3.402823e38;}
fn reflect101(x:i32)->i32 {if(x<0){return -x;}if(x>127){return 254-x;}return x;}
fn pi(q:u32,plane:u32,x:i32,y:i32)->u32{return q*81920u+plane*16384u+u32(y)*128u+u32(x);}
fn imageAt(x:i32,y:i32)->f32{return textureLoad(image,clamp(vec2<i32>(x,y),vec2<i32>(0),vec2<i32>(i32(p.width)-1,i32(p.height)-1)),0).r;}
fn imageSample(x:f32,y:f32)->f32{
 // OpenCV INTER_BITS=5 quantized interpolation, replicated source border.
 let sx=i32(round(x*32.0));let sy=i32(round(y*32.0));let ix=sx>>5;let iy=sy>>5;
 let a=f32(sx&31)/32.0;let b=f32(sy&31)/32.0;
 return (imageAt(ix,iy)*(1.0-a)+imageAt(ix+1,iy)*a)*(1.0-b)+(imageAt(ix,iy+1)*(1.0-a)+imageAt(ix+1,iy+1)*a)*b;
}
fn referenceValue(q:u32,x:i32,y:i32)->f32{return refs[u32(y/8)*16u+u32(x/8)][u32(state[q].rotation)]/255.0;}
fn rawAt(q:u32,x:i32,y:i32)->f32{return patches[pi(q,0u,reflect101(x),reflect101(y))];}
fn smoothed(q:u32,x:i32,y:i32,plane:u32)->f32{return patches[pi(q,plane,reflect101(x),reflect101(y))];}
fn patchAt(q:u32,plane:u32,x:i32,y:i32)->f32{if(x<0||x>127||y<0||y>127){return 0.0;}return patches[pi(q,plane,x,y)];}
fn patchSample(q:u32,plane:u32,x:f32,y:f32)->f32{
 let sx=i32(round(x*32.0));let sy=i32(round(y*32.0));let ix=sx>>5;let iy=sy>>5;
 let a=f32(sx&31)/32.0;let b=f32(sy&31)/32.0;
 return (patchAt(q,plane,ix,iy)*(1.0-a)+patchAt(q,plane,ix+1,iy)*a)*(1.0-b)+(patchAt(q,plane,ix,iy+1)*(1.0-a)+patchAt(q,plane,ix+1,iy+1)*a)*b;
}
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u){return;}let x=f32(k%128u);let y=f32(k/128u);let b=state[q].base;
 let den=b[6]*x+b[7]*y+b[8];let px=(b[0]*x+b[1]*y+b[2])/den;let py=(b[3]*x+b[4]*y+b[5])/den;
 patches[q*81920u+k]=imageSample(px,py);
}
@compute @workgroup_size(64) fn gaussian(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u){return;}let x=i32(k%128u);let y=i32(k/128u);var a=0.0;var b=0.0;
 for(var j=-1;j<=1;j++){for(var i=-1;i<=1;i++){let weight=select(1.0,2.0,i==0)*select(1.0,2.0,j==0)/16.0;a+=rawAt(q,x+i,y+j)*weight;b+=referenceValue(q,reflect101(x+i),reflect101(y+j))*weight;}}
 patches[pi(q,1u,x,y)]=a;patches[pi(q,2u,x,y)]=b;
}
@compute @workgroup_size(64) fn gradients(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u){return;}let x=i32(k%128u);let y=i32(k/128u);
 patches[pi(q,3u,x,y)]=.5*(smoothed(q,x+1,y,1u)-smoothed(q,x-1,y,1u));patches[pi(q,4u,x,y)]=.5*(smoothed(q,x,y+1,1u)-smoothed(q,x,y-1,1u));
}
@compute @workgroup_size(64) fn warp_pixels(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u||state[q].status!=0.0){return;}let x=f32(k%128u);let y=f32(k/128u);let w=state[q].warp;
 let d=w[6]*x+w[7]*y+1.0;let u=(w[0]*x+w[1]*y+w[2])/d;let v=(w[3]*x+w[4]*y+w[5])/d;
 let valid=finite(u)&&finite(v)&&abs(d)>0.0;let nx=i32(round(u));let ny=i32(round(v));let mask=valid&&nx>=0&&nx<128&&ny>=0&&ny<128;
 let at=(q*16384u+k)*5u;pixels[at]=patchSample(q,1u,u,v);pixels[at+1u]=patches[pi(q,2u,i32(k%128u),i32(k/128u))];pixels[at+2u]=patchSample(q,3u,u,v);pixels[at+3u]=patchSample(q,4u,u,v);pixels[at+4u]=select(0.0,1.0,mask);
}
@compute @workgroup_size(128) fn statistics(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let q=g.x;
 let is_active = q < p.count && state[q].status == 0.0;
 var s0=0.0; var s1=0.0; var s2=0.0; var s3=0.0; var s4=0.0; var s5=0.0;
 if(is_active){
  for(var k=lane;k<16384u;k+=128u){
   let a=(q*16384u+k)*5u;
   let im=pixels[a];let t=pixels[a+1u];let m=pixels[a+4u];
   s0+=m; s1+=im*m; s2+=t*m; s3+=im*im*m; s4+=t*t*m; s5+=im*t*m;
  }
 }
 var s = array<f32, 6>(s0, s1, s2, s3, s4, s5);
 for(var metric=0u; metric<6u; metric++){
  reduce[lane]=s[metric];
  workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){
   if(lane<stride){reduce[lane]+=reduce[lane+stride];}
   workgroupBarrier();
  }
  if(lane==0u && is_active){sums[q*86u+metric]=reduce[0];}
 }
}
@compute @workgroup_size(64) fn jacobian(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u||state[q].status!=0.0){return;}let a=(q*16384u+k)*5u;let x=f32(k%128u);let y=f32(k/128u);let w=state[q].warp;let den=w[6]*x+w[7]*y+1.0;
 let gx=pixels[a+2u]/den;let gy=pixels[a+3u]/den;let u=(w[0]*x+w[1]*y+w[2])/den;let v=(w[3]*x+w[4]*y+w[5])/den;let t= -u*gx-v*gy;
 let j=(q*16384u+k)*8u;jac[j]=gx*x;jac[j+1u]=gy*x;jac[j+2u]=t*x;jac[j+3u]=gx*y;jac[j+4u]=gy*y;jac[j+5u]=t*y;jac[j+6u]=gx;jac[j+7u]=gy;
 // OpenCV subtract(image,mean,image,mask) leaves unmasked image pixels intact.
 let n=sums[q*86u];let mi=sums[q*86u+1u]/n;let mt=sums[q*86u+2u]/n;
 pixels[a]=pixels[a]-mi*pixels[a+4u];pixels[a+1u]=(pixels[a+1u]-mt)*pixels[a+4u];
}
@compute @workgroup_size(128) fn normal_equations(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let q=g.x;
 let is_active = q < p.count && state[q].status == 0.0;
 var h: array<f32, 52>;
 if(is_active){
  for(var k=lane;k<16384u;k+=128u){
   let j=(q*16384u+k)*8u;let a=(q*16384u+k)*5u;
   let j0=jac[j]; let j1=jac[j+1u]; let j2=jac[j+2u]; let j3=jac[j+3u];
   let j4=jac[j+4u]; let j5=jac[j+5u]; let j6=jac[j+6u]; let j7=jac[j+7u];
   let jv = array<f32, 8>(j0, j1, j2, j3, j4, j5, j6, j7);
   let p0=pixels[a]; let p1=pixels[a+1u];
   var idx = 0u;
   for(var r=0u; r<8u; r++){
    for(var c=r; c<8u; c++){
     h[idx] += jv[r] * jv[c];
     idx++;
    }
   }
   for(var r=0u; r<8u; r++){
    h[36u + r] += jv[r] * p0;
    h[44u + r] += jv[r] * p1;
   }
  }
 }
 let o = q * 86u + 6u;
 var idx = 0u;
 for(var r=0u; r<8u; r++){
  for(var c=r; c<8u; c++){
   reduce[lane] = h[idx];
   workgroupBarrier();
   for(var stride=64u; stride>0u; stride/=2u){
    if(lane < stride){ reduce[lane] += reduce[lane + stride]; }
    workgroupBarrier();
   }
   if(lane == 0u && is_active){
    let val = reduce[0];
    sums[o + r * 8u + c] = val;
    sums[o + c * 8u + r] = val;
   }
   idx++;
  }
 }
 for(var r=0u; r<8u; r++){
  reduce[lane] = h[36u + r];
  workgroupBarrier();
  for(var stride=64u; stride>0u; stride/=2u){
   if(lane < stride){ reduce[lane] += reduce[lane + stride]; }
   workgroupBarrier();
  }
  if(lane == 0u && is_active){ sums[o + 64u + r] = reduce[0]; }
 }
 for(var r=0u; r<8u; r++){
  reduce[lane] = h[44u + r];
  workgroupBarrier();
  for(var stride=64u; stride>0u; stride/=2u){
   if(lane < stride){ reduce[lane] += reduce[lane + stride]; }
   workgroupBarrier();
  }
  if(lane == 0u && is_active){ sums[o + 72u + r] = reduce[0]; }
 }
}
@compute @workgroup_size(1) fn update(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.x;if(q>=p.count||state[q].status!=0.0){return;}let o=q*86u;let n=sums[o];if(state[q].iterations==0.0){state[q].firstMask=n;state[q].minMask=n;state[q].maxMask=n;}state[q].lastMask=n;state[q].minMask=min(state[q].minMask,n);state[q].maxMask=max(state[q].maxMask,n);let vi=sums[o+3u]-sums[o+1u]*sums[o+1u]/n;let vt=sums[o+4u]-sums[o+2u]*sums[o+2u]/n;let corr=sums[o+5u]-sums[o+1u]*sums[o+2u]/n;
 let rho=corr/sqrt(vi*vt);if(n==0.0||!finite(rho)){state[q].status=-1.0;return;}
 // Invert the 8x8 Hessian by pivoted elimination, no damping or threshold relaxation.
 var a:array<f32,128>;for(var r=0u;r<8u;r++){for(var c=0u;c<8u;c++){a[r*16u+c]=sums[o+6u+r*8u+c];a[r*16u+c+8u]=select(0.0,1.0,r==c);}}
 for(var c=0u;c<8u;c++){var pivot=c;for(var r=c+1u;r<8u;r++){if(abs(a[r*16u+c])>abs(a[pivot*16u+c])){pivot=r;}}
 if(a[pivot*16u+c]==0.0||!finite(a[pivot*16u+c])){state[q].status=-2.0;return;}
 if(pivot!=c){for(var j=0u;j<16u;j++){let v=a[c*16u+j];a[c*16u+j]=a[pivot*16u+j];a[pivot*16u+j]=v;}}
 let d=a[c*16u+c];for(var j=0u;j<16u;j++){a[c*16u+j]/=d;}
 for(var r=0u;r<8u;r++){if(r==c){continue;}let f=a[r*16u+c];for(var j=0u;j<16u;j++){a[r*16u+j]-=f*a[c*16u+j];}}
 }
 var hi:array<f32,8>;var dotI=0.0;var dotT=0.0;
 for(var r=0u;r<8u;r++){for(var c=0u;c<8u;c++){hi[r]+=a[r*16u+c+8u]*sums[o+70u+c];}dotI+=sums[o+70u+r]*hi[r];dotT+=sums[o+78u+r]*hi[r];}
 let ld=corr-dotT;let lambda=(vi-dotI)/ld;if(ld<=0.0||!finite(lambda)){state[q].status=-3.0;return;}
 var delta:array<f32,8>;for(var r=0u;r<8u;r++){for(var c=0u;c<8u;c++){delta[r]+=a[r*16u+c+8u]*(lambda*sums[o+78u+c]-sums[o+70u+c]);}if(!finite(delta[r])){state[q].status=-1.0;return;}}
 let order=array<u32,8>(0u,3u,6u,1u,4u,7u,2u,5u);for(var r=0u;r<8u;r++){state[q].warp[order[r]]+=delta[r];}
 state[q].iterations+=1.0;state[q].last=state[q].rho;state[q].rho=rho;
 if(state[q].iterations>=15.0&&rho<0.65){state[q].status=-1.0;return;}
 if(state[q].iterations>=25.0&&rho<0.75){state[q].status=-1.0;return;}
 // Native updates W on the final iteration, then tests rho change on next loop.
 if(abs(rho-state[q].last)<0.00001){state[q].pad=1.0;state[q].status=1.0;}else if(state[q].iterations>=60.0){state[q].pad=2.0;state[q].status=1.0;}
}
`;
