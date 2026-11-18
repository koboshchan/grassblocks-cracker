// src/detect/ecc-batch.ts
var ECC_QUAD_BYTES = { states: 28 * 4, patch: 81920 * 4, pixels: 16384 * 5 * 4, jac: 16384 * 8 * 4, sums: 86 * 4, readback: (28 + 86) * 4 };
var eccScratchBudget = (l) => Math.floor(Math.min(l.maxStorageBufferBindingSize, l.maxBufferSize) / 2);
function eccBatchCapacity(l, budget = eccScratchBudget(l), requested = Number.MAX_SAFE_INTEGER) {
  const bytes = Object.values(ECC_QUAD_BYTES).reduce((a, b) => a + b, 0);
  if (![budget, requested, l.maxStorageBufferBindingSize, l.maxBufferSize, l.maxComputeWorkgroupsPerDimension].every((v) => Number.isSafeInteger(v) && v >= 0) || budget < bytes + 16 || l.maxComputeWorkgroupsPerDimension < 256) return 0;
  const storage = Math.max(ECC_QUAD_BYTES.states, ECC_QUAD_BYTES.patch, ECC_QUAD_BYTES.pixels, ECC_QUAD_BYTES.jac, ECC_QUAD_BYTES.sums);
  const buffer = Math.max(storage, ECC_QUAD_BYTES.readback);
  return Math.max(0, Math.floor(Math.min(requested, (budget - 16) / bytes, l.maxStorageBufferBindingSize / storage, l.maxBufferSize / buffer, l.maxComputeWorkgroupsPerDimension)));
}
var EccAllocationError = class extends Error {
};
async function withEccAllocationRetry(pending, capacity, run, onRetry) {
  let n = Math.min(pending, capacity);
  if (n < 1) throw new EccAllocationError("ECC cannot allocate even one quad");
  for (; ; ) {
    try {
      return { count: n, value: await run(n) };
    } catch (e) {
      if (!(e instanceof EccAllocationError) || n === 1) throw e;
      const next = Math.max(1, Math.floor(n / 2));
      onRetry(n, next);
      n = next;
    }
  }
}
async function allocateEccBuffers(device, descriptors) {
  const owned = [];
  let thrown;
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  try {
    for (const d of descriptors) owned.push(device.createBuffer(d));
  } catch (e) {
    thrown = e;
  }
  const validationResult = device.popErrorScope(), oomResult = device.popErrorScope();
  const [validation, oom] = await Promise.all([validationResult, oomResult]);
  if (thrown || validation || oom) {
    owned.forEach((b) => b.destroy());
    if (validation) throw Error(validation.message);
    if (oom || thrown instanceof Error && (/out.of.memory|allocation/i.test(thrown.message) || thrown.name === "GPUOutOfMemoryError")) throw new EccAllocationError(oom?.message ?? String(thrown));
    throw thrown ?? Error(validation?.message ?? "ECC allocation failed");
  }
  return owned;
}
function eccThroughputPolicy(capacity) {
  if (!Number.isSafeInteger(capacity) || capacity < 1) throw Error("Invalid ECC capacity");
  const inFlight = Math.min(2, capacity);
  const batchSize = Math.max(1, Math.min(32, Math.floor(capacity / inFlight)));
  return { inFlight, batchSize };
}
async function runEccWindow(count, capacity, run) {
  if (!Number.isSafeInteger(count) || count < 0) throw Error("Invalid ECC pending count");
  const { inFlight, batchSize } = eccThroughputPolicy(capacity), out = new Array(count);
  let cursor = 0, failure;
  const lane = async () => {
    while (cursor < count && !failure) {
      const base = cursor, n = Math.min(batchSize, count - base);
      cursor += n;
      try {
        const value = await run(base, n);
        if (value.length !== n) throw Error("Invalid ECC batch length");
        for (let i = 0; i < n; i++) out[base + i] = value[i];
      } catch (e) {
        failure = e;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(inFlight, Math.ceil(count / batchSize)) }, lane));
  if (failure) throw failure;
  return out;
}

// src/detect/ecc-shader.ts
var eccShader = String.raw`
struct Params { width:u32,height:u32,count:u32,pad:u32 }
struct State { base:array<f32,9>, warp:array<f32,9>, rho:f32,last:f32,status:f32,iterations:f32,rotation:f32,pad:f32,firstMask:f32,lastMask:f32,minMask:f32,maxMask:f32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var image:texture_2d<f32>;
@group(0) @binding(2) var<storage,read> refs:array<vec4<f32>,256>;
@group(0) @binding(3) var<storage,read_write> state:array<State>;
// Patch planes raw,smoothed,referenceValue,gradientX,gradientY.
@group(0) @binding(4) var<storage,read_write> patches:array<f32>;
// Per-pixel warped input,referenceValue,gx,gy and mask; Jacobian stores compact (gx,gy,t).
@group(0) @binding(5) var<storage,read_write> pixels:array<f32>;
@group(0) @binding(6) var<storage,read_write> jac:array<f32>;
// 6 statistics then 64 H,8 imageProjection,8 templateProjection.
@group(0) @binding(7) var<storage,read_write> sums:array<f32>;
var<workgroup> wg_active:u32;
var<workgroup> reduce_buf:array<vec4<f32>,832>;
fn finite(v:f32)->bool{return abs(v)<=3.402823e38;}
fn reflect101(x:i32)->i32 {if(x<0){return -x;}if(x>127){return 254-x;}return x;}
fn pi(q:u32,plane:u32,x:i32,y:i32)->u32{return q*81920u+plane*16384u+u32(y)*128u+u32(x);}
fn imageAt(x:i32,y:i32)->f32{return textureLoad(image,clamp(vec2<i32>(x,y),vec2<i32>(0),vec2<i32>(i32(p.width)-1,i32(p.height)-1)),0).r;}
fn imageSample(x:f32,y:f32)->f32{
 // OpenCV INTER_BITS=5 quantized interpolation, replicated source border.
 let sx=i32(round(x*32.0));let sy=i32(round(y*32.0));let ix=sx>>5;let iy=sy>>5;
 let a=f32(sx&31)/32.0;let b=f32(sy&31)/32.0;
 let top=imageAt(ix,iy);let bot=imageAt(ix,iy+1);
 let r0=top+(imageAt(ix+1,iy)-top)*a;let r1=bot+(imageAt(ix+1,iy+1)-bot)*a;
 return r0+(r1-r0)*b;
}
fn referenceValueRot(rot:u32,x:i32,y:i32)->f32{return refs[u32(y>>3)*16u+u32(x>>3)][rot]/255.0;}
fn rawAt(q:u32,x:i32,y:i32)->f32{return patches[pi(q,0u,reflect101(x),reflect101(y))];}
fn smoothed(q:u32,x:i32,y:i32,plane:u32)->f32{return patches[pi(q,plane,reflect101(x),reflect101(y))];}
fn patchAt(q:u32,plane:u32,x:i32,y:i32)->f32{if(x<0||x>127||y<0||y>127){return 0.0;}return patches[pi(q,plane,x,y)];}
fn patchSample(q:u32,plane:u32,x:f32,y:f32)->f32{
 let sx=i32(round(x*32.0));let sy=i32(round(y*32.0));let ix=sx>>5;let iy=sy>>5;
 let a=f32(sx&31)/32.0;let b=f32(sy&31)/32.0;
 if(ix>=0&&ix<127&&iy>=0&&iy<127){
  let base=q*81920u+plane*16384u+u32(iy)*128u+u32(ix);
  let p00=patches[base];let p10=patches[base+1u];let p01=patches[base+128u];let p11=patches[base+129u];
  let r0=p00+(p10-p00)*a;let r1=p01+(p11-p01)*a;
  return r0+(r1-r0)*b;
 }
 return (patchAt(q,plane,ix,iy)*(1.0-a)+patchAt(q,plane,ix+1,iy)*a)*(1.0-b)+(patchAt(q,plane,ix,iy+1)*(1.0-a)+patchAt(q,plane,ix+1,iy+1)*a)*b;
}
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u){return;}let x=f32(k&127u);let y=f32(k>>7u);let b=state[q].base;
 let den=b[6]*x+b[7]*y+b[8];let px=(b[0]*x+b[1]*y+b[2])/den;let py=(b[3]*x+b[4]*y+b[5])/den;
 patches[q*81920u+k]=imageSample(px,py);
}
@compute @workgroup_size(64) fn gaussian(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u){return;}let x=i32(k&127u);let y=i32(k>>7u);let rot=u32(state[q].rotation);var a=0.0;var b=0.0;
 for(var j=-1;j<=1;j++){let ry=reflect101(y+j);let wj=select(0.0625,0.125,j==0);for(var i=-1;i<=1;i++){let rx=reflect101(x+i);let weight=select(wj,wj*2.0,i==0);a+=patches[pi(q,0u,rx,ry)]*weight;b+=referenceValueRot(rot,rx,ry)*weight;}}
 patches[pi(q,1u,x,y)]=a;patches[pi(q,2u,x,y)]=b;
}
@compute @workgroup_size(64) fn gradients(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u){return;}let x=i32(k&127u);let y=i32(k>>7u);
 patches[pi(q,3u,x,y)]=.5*(smoothed(q,x+1,y,1u)-smoothed(q,x-1,y,1u));patches[pi(q,4u,x,y)]=.5*(smoothed(q,x,y+1,1u)-smoothed(q,x,y-1,1u));
}
@compute @workgroup_size(64) fn warp_pixels(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u||state[q].status!=0.0){return;}let x=f32(k&127u);let y=f32(k>>7u);let w=state[q].warp;
 let d=w[6]*x+w[7]*y+1.0;let u=(w[0]*x+w[1]*y+w[2])/d;let v=(w[3]*x+w[4]*y+w[5])/d;
 let valid=finite(u)&&finite(v)&&abs(d)>0.0;let nx=i32(round(u));let ny=i32(round(v));let mask=valid&&nx>=0&&nx<128&&ny>=0&&ny<128;
 let at=(q*16384u+k)*5u;
 if(!mask){pixels[at]=0.0;pixels[at+1u]=0.0;pixels[at+2u]=0.0;pixels[at+3u]=0.0;pixels[at+4u]=0.0;return;}
 pixels[at]=patchSample(q,1u,u,v);pixels[at+1u]=patches[q*81920u+32768u+k];pixels[at+2u]=patchSample(q,3u,u,v);pixels[at+3u]=patchSample(q,4u,u,v);pixels[at+4u]=1.0;
}
@compute @workgroup_size(128) fn statistics(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let q=g.x;
 if(lane==0u){wg_active=select(0u,1u,q<p.count&&state[q].status==0.0);}
 if(workgroupUniformLoad(&wg_active)==0u){return;}
 var s0=0.0;var s1=0.0;var s2=0.0;var s3=0.0;var s4=0.0;var s5=0.0;
 let q_base=q*81920u;
 for(var k=lane;k<16384u;k+=128u){
  let a=q_base+k*5u;
  let m=pixels[a+4u];
  if(m==0.0){continue;}
  let im=pixels[a];let t=pixels[a+1u];
  s0+=1.0;s1+=im;s2+=t;s3+=im*im;s4+=t*t;s5+=im*t;
 }
 reduce_buf[lane]=vec4<f32>(s0,s1,s2,s3);
 reduce_buf[128u+lane]=vec4<f32>(s4,s5,0.0,0.0);
 workgroupBarrier();
 for(var stride=64u;stride>0u;stride/=2u){
  if(lane<stride){
   reduce_buf[lane]+=reduce_buf[lane+stride];
   reduce_buf[128u+lane]+=reduce_buf[128u+lane+stride];
  }
  workgroupBarrier();
 }
 if(lane==0u){
  let v0=reduce_buf[0];let v1=reduce_buf[128u];
  let o=q*86u;
  sums[o]=v0.x;sums[o+1u]=v0.y;sums[o+2u]=v0.z;sums[o+3u]=v0.w;sums[o+4u]=v1.x;sums[o+5u]=v1.y;
 }
}
@compute @workgroup_size(64) fn jacobian(@builtin(global_invocation_id) id:vec3<u32>){
 let q=id.y;let k=id.x;if(q>=p.count||k>=16384u||state[q].status!=0.0){return;}
 let a=(q*16384u+k)*5u;
 if(pixels[a+4u]==0.0){return;}
 let x=f32(k&127u);let y=f32(k>>7u);let w=state[q].warp;let den=w[6]*x+w[7]*y+1.0;
 let gx=pixels[a+2u]/den;let gy=pixels[a+3u]/den;let u=(w[0]*x+w[1]*y+w[2])/den;let v=(w[3]*x+w[4]*y+w[5])/den;let t= -u*gx-v*gy;
 let j=(q*16384u+k)*3u;jac[j]=gx;jac[j+1u]=gy;jac[j+2u]=t;
 let n=sums[q*86u];let mi=sums[q*86u+1u]/n;let mt=sums[q*86u+2u]/n;
 pixels[a]=pixels[a]-mi;pixels[a+1u]=pixels[a+1u]-mt;
}
@compute @workgroup_size(128) fn normal_equations(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let q=g.x;
 if(lane==0u){wg_active=select(0u,1u,q<p.count&&state[q].status==0.0);}
 if(workgroupUniformLoad(&wg_active)==0u){return;}
 var h0=vec4<f32>(0.0);var h1=vec4<f32>(0.0);var h2=vec4<f32>(0.0);var h3=vec4<f32>(0.0);
 var h4=vec4<f32>(0.0);var h5=vec4<f32>(0.0);var h6=vec4<f32>(0.0);var h7=vec4<f32>(0.0);
 var h8=vec4<f32>(0.0);var h9=vec4<f32>(0.0);var h10=vec4<f32>(0.0);var h11=vec4<f32>(0.0);
 var h12=vec4<f32>(0.0);
 let x=f32(lane);
 let pix_base=q*81920u;
 let jac_base=q*49152u;
 for(var k=lane;k<16384u;k+=128u){
  let a=pix_base+k*5u;
  if(pixels[a+4u]==0.0){continue;}
  let p0=pixels[a];let p1=pixels[a+1u];
  let jb=jac_base+k*3u;
  let gx=jac[jb];let gy=jac[jb+1u];let t=jac[jb+2u];
  if(gx==0.0&&gy==0.0&&p0==0.0&&p1==0.0){continue;}
  let y=f32(k>>7u);
  let j0=gx*x;let j1=gy*x;let j2=t*x;
  let j3=gx*y;let j4=gy*y;let j5=t*y;
  let j6=gx;let j7=gy;
  let jv0=vec4<f32>(j0,j1,j2,j3);
  let jv1=vec4<f32>(j4,j5,j6,j7);
  h0+=j0*jv0;
  h1+=j0*jv1;
  h2+=j1*vec4<f32>(j1,j2,j3,j4);
  h3+=vec4<f32>(j1*j5,j1*j6,j1*j7,j2*j2);
  h4+=j2*vec4<f32>(j3,j4,j5,j6);
  h5+=vec4<f32>(j2*j7,j3*j3,j3*j4,j3*j5);
  h6+=vec4<f32>(j3*j6,j3*j7,j4*j4,j4*j5);
  h7+=vec4<f32>(j4*j6,j4*j7,j5*j5,j5*j6);
  h8+=vec4<f32>(j5*j7,j6*j6,j6*j7,j7*j7);
  h9+=p0*jv0;
  h10+=p0*jv1;
  h11+=p1*jv0;
  h12+=p1*jv1;
 }
 if(lane>=64u){
  let b=(lane-64u)*13u;
  reduce_buf[b+0u]=h0;reduce_buf[b+1u]=h1;reduce_buf[b+2u]=h2;reduce_buf[b+3u]=h3;
  reduce_buf[b+4u]=h4;reduce_buf[b+5u]=h5;reduce_buf[b+6u]=h6;reduce_buf[b+7u]=h7;
  reduce_buf[b+8u]=h8;reduce_buf[b+9u]=h9;reduce_buf[b+10u]=h10;reduce_buf[b+11u]=h11;
  reduce_buf[b+12u]=h12;
 }
 workgroupBarrier();
 if(lane<64u){
  let b=lane*13u;
  reduce_buf[b+0u]=h0+reduce_buf[b+0u];reduce_buf[b+1u]=h1+reduce_buf[b+1u];
  reduce_buf[b+2u]=h2+reduce_buf[b+2u];reduce_buf[b+3u]=h3+reduce_buf[b+3u];
  reduce_buf[b+4u]=h4+reduce_buf[b+4u];reduce_buf[b+5u]=h5+reduce_buf[b+5u];
  reduce_buf[b+6u]=h6+reduce_buf[b+6u];reduce_buf[b+7u]=h7+reduce_buf[b+7u];
  reduce_buf[b+8u]=h8+reduce_buf[b+8u];reduce_buf[b+9u]=h9+reduce_buf[b+9u];
  reduce_buf[b+10u]=h10+reduce_buf[b+10u];reduce_buf[b+11u]=h11+reduce_buf[b+11u];
  reduce_buf[b+12u]=h12+reduce_buf[b+12u];
 }
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){
  let total=stride*13u;
  for(var idx=lane;idx<total;idx+=128u){
   reduce_buf[idx]+=reduce_buf[idx+total];
  }
  workgroupBarrier();
 }
 if(lane==0u){
  let r0=reduce_buf[0u];let r1=reduce_buf[1u];let r2=reduce_buf[2u];let r3=reduce_buf[3u];
  let r4=reduce_buf[4u];let r5=reduce_buf[5u];let r6=reduce_buf[6u];let r7=reduce_buf[7u];
  let r8=reduce_buf[8u];let r9=reduce_buf[9u];let r10=reduce_buf[10u];let r11=reduce_buf[11u];
  let r12=reduce_buf[12u];
  let o=q*86u+6u;
  sums[o+0u]=r0.x;sums[o+1u]=r0.y;sums[o+8u]=r0.y;sums[o+2u]=r0.z;sums[o+16u]=r0.z;sums[o+3u]=r0.w;sums[o+24u]=r0.w;
  sums[o+4u]=r1.x;sums[o+32u]=r1.x;sums[o+5u]=r1.y;sums[o+40u]=r1.y;sums[o+6u]=r1.z;sums[o+48u]=r1.z;sums[o+7u]=r1.w;sums[o+56u]=r1.w;
  sums[o+9u]=r2.x;sums[o+10u]=r2.y;sums[o+17u]=r2.y;sums[o+11u]=r2.z;sums[o+25u]=r2.z;sums[o+12u]=r2.w;sums[o+33u]=r2.w;
  sums[o+13u]=r3.x;sums[o+41u]=r3.x;sums[o+14u]=r3.y;sums[o+49u]=r3.y;sums[o+15u]=r3.z;sums[o+57u]=r3.z;sums[o+18u]=r3.w;
  sums[o+19u]=r4.x;sums[o+26u]=r4.x;sums[o+20u]=r4.y;sums[o+34u]=r4.y;sums[o+21u]=r4.z;sums[o+42u]=r4.z;sums[o+22u]=r4.w;sums[o+50u]=r4.w;
  sums[o+23u]=r5.x;sums[o+58u]=r5.x;sums[o+27u]=r5.y;sums[o+28u]=r5.z;sums[o+35u]=r5.z;sums[o+29u]=r5.w;sums[o+43u]=r5.w;
  sums[o+30u]=r6.x;sums[o+51u]=r6.x;sums[o+31u]=r6.y;sums[o+59u]=r6.y;sums[o+36u]=r6.z;sums[o+37u]=r6.w;sums[o+44u]=r6.w;
  sums[o+38u]=r7.x;sums[o+52u]=r7.x;sums[o+39u]=r7.y;sums[o+60u]=r7.y;sums[o+45u]=r7.z;sums[o+46u]=r7.w;sums[o+53u]=r7.w;
  sums[o+47u]=r8.x;sums[o+61u]=r8.x;sums[o+54u]=r8.y;sums[o+55u]=r8.z;sums[o+62u]=r8.z;sums[o+63u]=r8.w;
  sums[o+64u]=r9.x;sums[o+65u]=r9.y;sums[o+66u]=r9.z;sums[o+67u]=r9.w;
  sums[o+68u]=r10.x;sums[o+69u]=r10.y;sums[o+70u]=r10.z;sums[o+71u]=r10.w;
  sums[o+72u]=r11.x;sums[o+73u]=r11.y;sums[o+74u]=r11.z;sums[o+75u]=r11.w;
  sums[o+76u]=r12.x;sums[o+77u]=r12.y;sums[o+78u]=r12.z;sums[o+79u]=r12.w;
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
 var delta:array<f32,8>;for(var r=0u;r<8u;r++){var ht=0.0;for(var c=0u;c<8u;c++){ht+=a[r*16u+c+8u]*sums[o+78u+c];}delta[r]=lambda*ht-hi[r];if(!finite(delta[r])){state[q].status=-1.0;return;}}
 let order=array<u32,8>(0u,3u,6u,1u,4u,7u,2u,5u);for(var r=0u;r<8u;r++){state[q].warp[order[r]]+=delta[r];}
 state[q].iterations+=1.0;state[q].last=state[q].rho;state[q].rho=rho;
 if(state[q].iterations>=15.0&&rho<0.65){state[q].status=-1.0;return;}
 if(state[q].iterations>=25.0&&rho<0.75){state[q].status=-1.0;return;}
 // Native updates W on the final iteration, then tests rho change on next loop.
 if(abs(rho-state[q].last)<0.00001){state[q].pad=1.0;state[q].status=1.0;}else if(state[q].iterations>=60.0){state[q].pad=2.0;state[q].status=1.0;}
}
`;

// src/detect/ecc.ts
var entries = ["prepare", "gaussian", "gradients", "warp_pixels", "statistics", "jacobian", "normal_equations", "update"];
var BatchedGpuEcc = class _BatchedGpuEcc {
  constructor(device, pipelines, querySet, resolveBuffer, queryReadbackBuffer) {
    this.device = device;
    this.pipelines = pipelines;
    this.capacity = eccBatchCapacity(device.limits);
    if (this.capacity < 1) throw Error("ECC device limits exceed conservative scratch budget");
    this.querySet = querySet;
    this.resolveBuffer = resolveBuffer;
    this.queryReadbackBuffer = queryReadbackBuffer;
    if (!this.querySet && device.features?.has("timestamp-query")) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
      this.resolveBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.queryReadbackBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    } else if (this.querySet && this.resolveBuffer && !this.queryReadbackBuffer) {
      this.queryReadbackBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    if (this.queryReadbackBuffer) {
      this.allQueryReadbacks.push(this.queryReadbackBuffer);
      this.freeQueryReadbacks.push(this.queryReadbackBuffer);
    }
  }
  device;
  pipelines;
  capacity;
  lastHostTiming = null;
  scratchSlots = [];
  querySet;
  resolveBuffer;
  queryReadbackBuffer;
  freeQueryReadbacks = [];
  allQueryReadbacks = [];
  // Each in-flight run needs its own timestamp staging buffer; a buffer with a pending mapAsync cannot be submitted.
  acquireQueryReadback() {
    if (!this.querySet || !this.resolveBuffer) return void 0;
    let b = this.freeQueryReadbacks.pop();
    if (!b) {
      b = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.allQueryReadbacks.push(b);
    }
    return b;
  }
  static async create(device) {
    const shader = device.createShaderModule({ code: eccShader });
    const info = await shader.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) throw Error(errors.map((m) => `${m.lineNum} ${m.message}`).join("\n"));
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      ...[2, 3, 4, 5, 6, 7].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 2 ? "read-only-storage" : "storage" } }))
    ] });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = await Promise.all(entries.map((entryPoint) => device.createComputePipelineAsync({ layout: pl, compute: { module: shader, entryPoint } })));
    let querySet;
    let resolveBuffer;
    let queryReadbackBuffer;
    if (device.features?.has("timestamp-query")) {
      querySet = device.createQuerySet({ type: "timestamp", count: 2 });
      resolveBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      queryReadbackBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    return new _BatchedGpuEcc(device, pipelines, querySet, resolveBuffer, queryReadbackBuffer);
  }
  async run(image, refs, width, height, quads, rotations, diagnosticInitialWarp, onTiming) {
    if (quads.length > this.capacity || quads.length !== rotations.length || quads.some((q) => q.length !== 4 || q.some((p) => p.length !== 2 || p.some((v) => !Number.isFinite(v)))) || rotations.some((r) => !Number.isInteger(r) || r < 0 || r > 3)) throw Error("ECC component bounds/inputs");
    if (diagnosticInitialWarp && (diagnosticInitialWarp.length !== 9 || diagnosticInitialWarp.some((v) => !Number.isFinite(v)) || diagnosticInitialWarp[8] !== 1)) throw Error("Diagnostic initial warp invalid");
    const wallStart = performance.now();
    const count = quads.length;
    if (!count) return [];
    const initial = new Float32Array(count * 28);
    quads.forEach((q, i) => {
      const base = quadBase(q);
      const s = i * 28;
      initial.set(base, s);
      initial.set(diagnosticInitialWarp ?? [1, 0, 0, 0, 1, 0, 0, 0, 1], s + 9);
      initial[s + 18] = -1;
      initial[s + 19] = -1e-5;
      initial[s + 20] = base.every(Number.isFinite) ? 0 : -2;
      initial[s + 22] = rotations[i];
    });
    const rw = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const slotSize = Math.max(1, Math.min(512, Math.floor(this.capacity / Math.min(2, this.capacity))));
    let slot = count <= slotSize ? this.scratchSlots.find((s) => !s.busy && s.capacity >= count) : void 0;
    let reusedBuffers = !!slot;
    if (slot) slot.busy = true;
    else if (count <= slotSize && this.scratchSlots.length < Math.min(2, this.capacity)) {
      slot = { capacity: slotSize, buffers: [], busy: true };
      this.scratchSlots.push(slot);
    }
    let owned;
    try {
      if (slot?.buffers.length) owned = slot.buffers;
      else {
        const n = slot?.capacity ?? count;
        const allocationSizes = [16, n * 28 * 4, n * 81920 * 4, n * 16384 * 5 * 4, n * 16384 * 8 * 4, n * 86 * 4, n * (28 + 86) * 4];
        owned = await allocateEccBuffers(this.device, allocationSizes.map((size, i) => ({ size, usage: i === 0 ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST : i === 6 ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST : rw })));
        if (slot) slot.buffers = owned;
      }
    } catch (e) {
      if (slot) {
        this.scratchSlots.splice(this.scratchSlots.indexOf(slot), 1);
      }
      throw e;
    }
    const [params, states, patch, pixels, jac, sums, read] = owned;
    let mapped = false;
    let queryMapped = false;
    const qrb = this.acquireQueryReadback();
    try {
      this.device.queue.writeBuffer(params, 0, new Uint32Array([width, height, count, 0]));
      this.device.queue.writeBuffer(states, 0, initial);
      const bind = this.device.createBindGroup({ layout: this.pipelines[0].getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: image }, { binding: 2, resource: { buffer: refs } }, ...[states, patch, pixels, jac, sums].map((b, i) => ({ binding: i + 3, resource: { buffer: b } }))] });
      const encodeStart = performance.now();
      const encoder = this.device.createCommandEncoder();
      for (const buffer of [patch, pixels, jac, sums]) encoder.clearBuffer(buffer);
      const hasTimestamps = Boolean(this.querySet && this.resolveBuffer && qrb);
      const compute = hasTimestamps ? encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }) : encoder.beginComputePass();
      compute.setBindGroup(0, bind);
      const pass = (index, x, y = 1) => {
        compute.setPipeline(this.pipelines[index]);
        compute.dispatchWorkgroups(x, y);
      };
      pass(0, 256, count);
      pass(1, 256, count);
      pass(2, 256, count);
      for (let iteration = 0; iteration < 60; iteration++) {
        pass(3, 256, count);
        pass(4, count, 1);
        pass(5, 256, count);
        pass(6, count, 1);
        pass(7, count);
      }
      compute.end();
      if (hasTimestamps) {
        encoder.resolveQuerySet(this.querySet, 0, 2, this.resolveBuffer, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer, 0, qrb, 0, 16);
      }
      encoder.copyBufferToBuffer(states, 0, read, 0, initial.byteLength);
      encoder.copyBufferToBuffer(sums, 0, read, initial.byteLength, count * 86 * 4);
      const command = encoder.finish(), encoded = performance.now();
      this.device.queue.submit([command]);
      let gpuTimestampMs = null;
      if (hasTimestamps) {
        await Promise.all([read.mapAsync(GPUMapMode.READ), qrb.mapAsync(GPUMapMode.READ)]);
        mapped = true;
        queryMapped = true;
        const ts = new BigInt64Array(qrb.getMappedRange());
        const start = ts[0], end = ts[1];
        gpuTimestampMs = Number(end - start) / 1e6;
      } else {
        await read.mapAsync(GPUMapMode.READ);
        mapped = true;
      }
      this.lastHostTiming = { count, passes: 1, dispatches: 303, submissions: 1, readbacks: 1, encodeMs: encoded - encodeStart, submitToReadbackMs: performance.now() - encoded, wallMs: performance.now() - wallStart, scratchBytes: count * Object.values(ECC_QUAD_BYTES).reduce((a, b) => a + b, 0) + 16, gpuTimestampMs, reusedBuffers };
      onTiming?.(this.lastHostTiming);
      const raw = new Float32Array(read.getMappedRange()).slice();
      return quads.map((q, i) => {
        const s = i * 28, base = Array.from(raw.slice(s, s + 9)), warp = Array.from(raw.slice(s + 9, s + 18));
        const composed = multiply(base, warp);
        const candidate = [[0, 0], [128, 0], [128, 128], [0, 128]].map(([x, y]) => {
          const d = composed[6] * x + composed[7] * y + composed[8];
          return [(composed[0] * x + composed[1] * y + composed[2]) / d, (composed[3] * x + composed[4] * y + composed[5]) / d];
        });
        return { base, warp, composed, rho: raw[s + 18], iterations: raw[s + 21], status: raw[s + 20], lastRho: raw[s + 19], rhoChange: Math.abs(raw[s + 18] - raw[s + 19]), converged: raw[s + 23] === 1, iterationLimitReached: raw[s + 23] === 2, termination: raw[s + 20] < 0 ? "failure" : raw[s + 23] === 1 ? "epsilon" : raw[s + 23] === 2 ? "iteration-limit" : "unfinished", maskCount: raw[s + 25], firstMaskCount: raw[s + 24], minMaskCount: raw[s + 26], maxMaskCount: raw[s + 27], rotation: rotations[i], quad: raw[s + 20] > 0 ? candidate : q };
      });
    } finally {
      if (mapped) read.unmap();
      if (queryMapped) qrb?.unmap();
      if (qrb) this.freeQueryReadbacks.push(qrb);
      if (slot) slot.busy = false;
      else owned.forEach((b) => b.destroy());
    }
  }
  destroy() {
    for (const slot of this.scratchSlots) slot.buffers.forEach((b) => b.destroy());
    this.scratchSlots = [];
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    this.allQueryReadbacks.forEach((b) => b.destroy());
    this.allQueryReadbacks = [];
    this.freeQueryReadbacks = [];
    this.querySet = void 0;
    this.resolveBuffer = void 0;
    this.queryReadbackBuffer = void 0;
  }
};
function quadBase(q) {
  const [a, b, c, d] = q;
  const dx1 = b[0] - c[0], dx2 = d[0] - c[0], dx3 = a[0] - b[0] + c[0] - d[0], dy1 = b[1] - c[1], dy2 = d[1] - c[1], dy3 = a[1] - b[1] + c[1] - d[1];
  const det = dx1 * dy2 - dx2 * dy1;
  const g = (dx3 * dy2 - dx2 * dy3) / det / 128, h = (dx1 * dy3 - dx3 * dy1) / det / 128;
  return [(b[0] - a[0]) / 128 + g * b[0], (d[0] - a[0]) / 128 + h * d[0], a[0], (b[1] - a[1]) / 128 + g * b[1], (d[1] - a[1]) / 128 + h * d[1], a[1], g, h, 1];
}
function multiply(a, b) {
  return Array.from({ length: 9 }, (_, i) => [0, 1, 2].reduce((s, k) => s + a[Math.floor(i / 3) * 3 + k] * b[k * 3 + i % 3], 0));
}

// src/detect/webgpu.ts
var WebGpuDetector = class _WebGpuDetector {
  adapter;
  device;
  pipeline;
  refinePipeline = null;
  imageTexture = null;
  imageView = null;
  sampler = null;
  imageWidth = 0;
  imageHeight = 0;
  lastImageData = null;
  paramsBuffer;
  refStatsBuffer = null;
  refsBuffer = null;
  paramsData = new Uint32Array(4);
  refStatsData = null;
  refPatternData = null;
  quadsBuffer = null;
  quadsBufferCapacity = 0;
  resultsBuffer = null;
  readBuffer = null;
  resultsBufferCapacity = 0;
  classifyPoolBusy = false;
  cachedBindGroup = null;
  bindGroupImageView = null;
  destroyed = false;
  constructor(adapter, device, pipeline, refinePipeline = null) {
    this.adapter = adapter;
    this.device = device;
    this.pipeline = pipeline;
    this.refinePipeline = refinePipeline;
    this.device.lost.then((info) => {
      console.warn(`WebGPU device lost: ${info.message}`);
      this.destroyed = true;
    });
    this.paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });
  }
  /** Surface async validation/device errors before returning any GPU result. */
  async checked(work) {
    if (this.destroyed) throw Error("WebGPU device unavailable");
    const device = this.device;
    device.pushErrorScope("internal");
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    try {
      const lost = device.lost.then((info) => {
        throw Error("WebGPU device lost: " + info.message);
      });
      const value = await Promise.race([work(), lost]);
      await Promise.race([device.queue.onSubmittedWorkDone(), lost]);
      return value;
    } finally {
      const errors = await Promise.all([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
      const error = errors.find(Boolean);
      if (error) throw Error("WebGPU execution failed: " + error.message);
    }
  }
  static async isSupported() {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      return false;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }
  static async create(shaderCode) {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      return null;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) {
        return null;
      }
      const requiredLimits = {};
      for (const key of ["maxStorageBufferBindingSize", "maxBufferSize", "maxComputeWorkgroupsPerDimension"]) {
        const value = adapter.limits?.[key];
        if (Number.isSafeInteger(value) && value > 0) requiredLimits[key] = value;
      }
      let device;
      if (adapter.features?.has("timestamp-query")) {
        try {
          device = await adapter.requestDevice({
            requiredLimits,
            requiredFeatures: ["timestamp-query"]
          });
        } catch {
          device = await adapter.requestDevice({ requiredLimits }).catch(() => adapter.requestDevice());
        }
      } else {
        device = await adapter.requestDevice({ requiredLimits }).catch(() => adapter.requestDevice());
      }
      const shaderModule = device.createShaderModule({ code: shaderCode });
      const compilationInfo = await shaderModule.getCompilationInfo();
      const errors = compilationInfo.messages.filter((m) => m.type === "error");
      if (errors.length > 0) {
        device.destroy();
        throw new Error(`Shader compilation failed:
${errors.map((e) => e.message).join("\n")}`);
      }
      const match = shaderCode.match(/@compute[\s\S]*?fn\s+([a-zA-Z0-9_]+)\s*\(/);
      const entryPoint = match ? match[1] : "classify_main";
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: shaderModule,
          entryPoint
        }
      });
      let refinePipeline = null;
      if (shaderCode.includes("refine_main")) {
        refinePipeline = await device.createComputePipelineAsync({
          layout: "auto",
          compute: {
            module: shaderModule,
            entryPoint: "refine_main"
          }
        });
      }
      return new _WebGpuDetector(adapter, device, pipeline, refinePipeline);
    } catch (err) {
      console.warn("Failed to create WebGpuDetector:", err);
      return null;
    }
  }
  setImage(rgba, width, height) {
    if (this.destroyed) throw new Error("WebGpuDetector is destroyed");
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2) {
      throw new Error(`Invalid image dimensions: ${width}x${height}`);
    }
    if (this.imageTexture && this.imageWidth === width && this.imageHeight === height && this.lastImageData === rgba) {
      return;
    }
    if (this.imageTexture) {
      this.imageTexture.destroy();
      this.imageTexture = null;
      this.imageView = null;
    }
    this.imageWidth = width;
    this.imageHeight = height;
    this.lastImageData = rgba;
    this.imageTexture = this.device.createTexture({
      size: [width, height, 1],
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    this.imageView = this.imageTexture.createView();
    const isRgba = rgba.length >= width * height * 4;
    const alignedBytesPerRow = Math.ceil(width / 256) * 256;
    let textureData;
    if (alignedBytesPerRow === width && !isRgba) {
      textureData = rgba;
    } else {
      textureData = new Uint8Array(alignedBytesPerRow * height);
      for (let y = 0; y < height; y++) {
        const srcRow = y * width;
        const dstRow = y * alignedBytesPerRow;
        if (isRgba) {
          for (let x = 0; x < width; x++) {
            textureData[dstRow + x] = rgba[(srcRow + x) * 4 + 1];
          }
        } else {
          textureData.set(rgba.subarray(srcRow, srcRow + width), dstRow);
        }
      }
    }
    this.device.queue.writeTexture(
      { texture: this.imageTexture },
      textureData,
      { bytesPerRow: alignedBytesPerRow, rowsPerImage: height },
      { width, height }
    );
  }
  setRefs(refs) {
    if (this.destroyed) throw new Error("WebGpuDetector is destroyed");
    const refPatternData = this.refPatternData ??= new Float32Array(256 * 4);
    if (refs.length === 16 && refs[0].length === 16) {
      for (let y = 0; y < 16; y++) {
        const row = refs[y];
        const yRot2 = 15 - y;
        for (let x = 0; x < 16; x++) {
          const idx = y << 4 | x;
          const base = idx << 2;
          refPatternData[base + 0] = row[x];
          refPatternData[base + 1] = refs[15 - x][y];
          refPatternData[base + 2] = refs[yRot2][15 - x];
          refPatternData[base + 3] = refs[x][yRot2];
        }
      }
    } else if (refs.length === 4 && refs[0].length === 256) {
      const r0 = refs[0];
      const r1 = refs[1];
      const r2 = refs[2];
      const r3 = refs[3];
      for (let idx = 0; idx < 256; idx++) {
        const base = idx << 2;
        refPatternData[base + 0] = r0[idx];
        refPatternData[base + 1] = r1[idx];
        refPatternData[base + 2] = r2[idx];
        refPatternData[base + 3] = r3[idx];
      }
    } else {
      throw new Error("Expected 4 flat 16x16 rotations or a 16x16 reference matrix");
    }
    const refStatsData = this.refStatsData ??= new Float32Array(64);
    refStatsData.fill(0);
    for (let j = 1; j < 15; j++) {
      const qRow = j >= 8 ? 8 : 0;
      for (let i = 1; i < 15; i++) {
        const qOffset = (i >= 8 ? 4 : 0) + qRow;
        const base = (j << 4 | i) << 2;
        const qSb = 12 + qOffset;
        const qBb = 28 + qOffset;
        const v0 = refPatternData[base + 0];
        const v1 = refPatternData[base + 1];
        const v2 = refPatternData[base + 2];
        const v3 = refPatternData[base + 3];
        refStatsData[0] += v0;
        refStatsData[1] += v1;
        refStatsData[2] += v2;
        refStatsData[3] += v3;
        refStatsData[4] += v0 * v0;
        refStatsData[5] += v1 * v1;
        refStatsData[6] += v2 * v2;
        refStatsData[7] += v3 * v3;
        refStatsData[qSb + 0] += v0;
        refStatsData[qSb + 1] += v1;
        refStatsData[qSb + 2] += v2;
        refStatsData[qSb + 3] += v3;
        refStatsData[qBb + 0] += v0 * v0;
        refStatsData[qBb + 1] += v1 * v1;
        refStatsData[qBb + 2] += v2 * v2;
        refStatsData[qBb + 3] += v3 * v3;
      }
    }
    for (let k = 0; k < 4; k++) {
      const sbVal = refStatsData[k];
      refStatsData[8 + k] = refStatsData[4 + k] - sbVal * sbVal / 196;
    }
    for (let q = 0; q < 4; q++) {
      const qOffset = q << 2;
      for (let k = 0; k < 4; k++) {
        const qSbVal = refStatsData[12 + qOffset + k];
        refStatsData[44 + qOffset + k] = refStatsData[28 + qOffset + k] - qSbVal * qSbVal / 49;
      }
    }
    if (!this.refStatsBuffer) {
      this.refStatsBuffer = this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    this.device.queue.writeBuffer(this.refStatsBuffer, 0, refStatsData);
    if (!this.refsBuffer) {
      this.refsBuffer = this.device.createBuffer({
        size: 256 * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    this.device.queue.writeBuffer(this.refsBuffer, 0, refPatternData);
  }
  async classifyBatch(quads) {
    const classifiedAt = performance.now();
    this.diagnostics.classified += quads.length / 8;
    if (this.destroyed) throw new Error("WebGpuDetector is destroyed");
    if (!this.imageTexture || !this.imageView || !this.sampler) {
      throw new Error("Image not set. Call setImage() before classifyBatch()");
    }
    if (!this.refStatsBuffer || !this.refsBuffer) {
      throw new Error("References not set. Call setRefs() before classifyBatch()");
    }
    if (quads.length % 8 !== 0) {
      throw new Error("Quads length must be a multiple of 8 (8 floats per quad)");
    }
    const count = quads.length / 8;
    if (count === 0) {
      return new Float32Array(0);
    }
    this.paramsData[0] = this.imageWidth;
    this.paramsData[1] = this.imageHeight;
    this.paramsData[2] = count;
    this.paramsData[3] = 0;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
    const quadsByteLength = Math.max(32, quads.byteLength);
    const resultsByteLength = Math.max(20, count * 20);
    const usePool = !this.classifyPoolBusy;
    if (usePool) {
      this.classifyPoolBusy = true;
    }
    let quadsBuffer = null;
    let resultsBuffer = null;
    let readBuffer = null;
    let mapped = false;
    try {
      let bindGroup;
      if (usePool) {
        if (!this.quadsBuffer || this.quadsBufferCapacity < quadsByteLength) {
          this.quadsBuffer?.destroy();
          this.quadsBuffer = this.device.createBuffer({
            size: quadsByteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
          });
          this.quadsBufferCapacity = quadsByteLength;
          this.cachedBindGroup = null;
        }
        if (!this.resultsBuffer || !this.readBuffer || this.resultsBufferCapacity < resultsByteLength) {
          this.resultsBuffer?.destroy();
          this.readBuffer?.destroy();
          this.resultsBuffer = this.device.createBuffer({
            size: resultsByteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
          });
          this.readBuffer = this.device.createBuffer({
            size: resultsByteLength,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
          });
          this.resultsBufferCapacity = resultsByteLength;
          this.cachedBindGroup = null;
        }
        quadsBuffer = this.quadsBuffer;
        resultsBuffer = this.resultsBuffer;
        readBuffer = this.readBuffer;
        if (!this.cachedBindGroup || this.bindGroupImageView !== this.imageView) {
          this.cachedBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.paramsBuffer } },
              { binding: 1, resource: { buffer: this.refStatsBuffer } },
              { binding: 2, resource: { buffer: quadsBuffer } },
              { binding: 3, resource: { buffer: resultsBuffer } },
              { binding: 4, resource: this.imageView },
              { binding: 5, resource: this.sampler },
              { binding: 6, resource: { buffer: this.refsBuffer } }
            ]
          });
          this.bindGroupImageView = this.imageView;
        }
        bindGroup = this.cachedBindGroup;
      } else {
        quadsBuffer = this.device.createBuffer({
          size: quadsByteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        resultsBuffer = this.device.createBuffer({
          size: resultsByteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        readBuffer = this.device.createBuffer({
          size: resultsByteLength,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        bindGroup = this.device.createBindGroup({
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.paramsBuffer } },
            { binding: 1, resource: { buffer: this.refStatsBuffer } },
            { binding: 2, resource: { buffer: quadsBuffer } },
            { binding: 3, resource: { buffer: resultsBuffer } },
            { binding: 4, resource: this.imageView },
            { binding: 5, resource: this.sampler },
            { binding: 6, resource: { buffer: this.refsBuffer } }
          ]
        });
      }
      this.device.queue.writeBuffer(quadsBuffer, 0, quads);
      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      const workgroupCount = Math.ceil(count / 64);
      pass.dispatchWorkgroups(workgroupCount, 1, 1);
      pass.end();
      commandEncoder.copyBufferToBuffer(resultsBuffer, 0, readBuffer, 0, count * 20);
      this.device.queue.submit([commandEncoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      mapped = true;
      const out = new Float32Array(count * 5);
      out.set(new Float32Array(readBuffer.getMappedRange(0, count * 20)));
      return out;
    } finally {
      if (mapped && readBuffer) {
        readBuffer.unmap();
      }
      this.diagnostics.classifyMs += performance.now() - classifiedAt;
      if (usePool) {
        this.classifyPoolBusy = false;
      } else {
        quadsBuffer?.destroy();
        resultsBuffer?.destroy();
        readBuffer?.destroy();
      }
    }
  }
  eccComponent = null;
  diagnostics = { classified: 0, eccCandidates: 0, classifyMs: 0, eccMs: 0, eccIterations: 0, eccConverged: 0, eccIterationLimits: 0, eccFailures: 0 };
  get adapterInfo() {
    const info = this.adapter.info;
    return info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: this.adapter.isFallbackAdapter } : { description: "adapter info unavailable" };
  }
  eccBatchPolicy = null;
  eccBatchTimings = [];
  async refineBatch(quads, rotations) {
    if (this.destroyed || !this.imageView || !this.refsBuffer) throw Error("ECC image/references unavailable");
    this.eccComponent ??= BatchedGpuEcc.create(this.device);
    const component = await this.eccComponent;
    const count = quads.length / 8;
    if (!Number.isInteger(count)) throw Error("Invalid ECC quad buffer");
    if (!rotations) {
      const v = await this.classifyBatch(quads);
      rotations = new Array(count);
      for (let i = 0; i < count; i++) {
        rotations[i] = Math.round(v[i * 5 + 1]);
      }
    }
    const output = new Float32Array(quads.length);
    const start = performance.now();
    const policy = eccThroughputPolicy(component.capacity);
    const allocationFallbacks = [];
    this.eccBatchPolicy = { capacity: component.capacity, chosenBatchSize: Math.min(count, policy.batchSize), inFlight: policy.inFlight, workingSetPolicy: "device-scaled batching up to capacity, two independent batches in flight", scratchBudgetBytes: eccScratchBudget(this.device.limits), bytesPerQuad: 1180560, allocationFallbacks, limits: { maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize, maxBufferSize: this.device.limits.maxBufferSize, maxComputeWorkgroupsPerDimension: this.device.limits.maxComputeWorkgroupsPerDimension } };
    const traces = await runEccWindow(count, component.capacity, async (base, n) => {
      const batch = [];
      for (let offset = 0; offset < n; ) {
        const result = await withEccAllocationRetry(n - offset, n - offset, async (size) => {
          const qs = new Array(size);
          for (let j = 0; j < size; j++) {
            const idx = (base + offset + j) * 8;
            qs[j] = [
              [quads[idx], quads[idx + 1]],
              [quads[idx + 2], quads[idx + 3]],
              [quads[idx + 4], quads[idx + 5]],
              [quads[idx + 6], quads[idx + 7]]
            ];
          }
          return component.run(this.imageView, this.refsBuffer, this.imageWidth, this.imageHeight, qs, rotations.slice(base + offset, base + offset + size), void 0, (timing) => this.eccBatchTimings.push({ ...timing }));
        }, (from, to) => allocationFallbacks.push({ from, to }));
        batch.push(...result.value);
        offset += result.count;
      }
      return batch;
    });
    const sizes = this.eccBatchTimings.map((t) => t.count);
    this.eccBatchPolicy.batchHistogram = Object.fromEntries([...new Set(sizes)].map((n) => [n, sizes.filter((x) => x === n).length]));
    this.eccBatchPolicy.actualBatchMin = sizes.length ? Math.min(...sizes) : 0;
    this.eccBatchPolicy.actualBatchMax = sizes.length ? Math.max(...sizes) : 0;
    this.eccBatchPolicy.maxWorkingSetBytes = policy.batchSize * policy.inFlight * 1180560 + policy.inFlight * 16;
    this.eccBatchPolicy.reusedBatches = this.eccBatchTimings.filter((t) => t.reusedBuffers).length;
    for (let j = 0; j < traces.length; j++) {
      const t = traces[j];
      const q = t.quad;
      const dst = j * 8;
      output[dst] = q[0][0];
      output[dst + 1] = q[0][1];
      output[dst + 2] = q[1][0];
      output[dst + 3] = q[1][1];
      output[dst + 4] = q[2][0];
      output[dst + 5] = q[2][1];
      output[dst + 6] = q[3][0];
      output[dst + 7] = q[3][1];
      this.diagnostics.eccIterations += t.iterations;
      this.diagnostics.eccConverged += Number(t.converged);
      this.diagnostics.eccIterationLimits += Number(t.iterationLimitReached);
      this.diagnostics.eccFailures += Number(t.status < 0);
    }
    this.diagnostics.eccCandidates += count;
    this.diagnostics.eccMs += performance.now() - start;
    return output;
  }
  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    void this.eccComponent?.then((component) => component.destroy()).catch(() => {
    });
    this.paramsBuffer?.destroy();
    this.refStatsBuffer?.destroy();
    this.refsBuffer?.destroy();
    this.quadsBuffer?.destroy();
    this.resultsBuffer?.destroy();
    this.readBuffer?.destroy();
    this.imageTexture?.destroy();
    this.device?.destroy();
    this.refStatsBuffer = null;
    this.refsBuffer = null;
    this.quadsBuffer = null;
    this.quadsBufferCapacity = 0;
    this.resultsBuffer = null;
    this.readBuffer = null;
    this.resultsBufferCapacity = 0;
    this.classifyPoolBusy = false;
    this.cachedBindGroup = null;
    this.bindGroupImageView = null;
    this.refStatsData = null;
    this.refPatternData = null;
    this.imageTexture = null;
    this.imageView = null;
    this.lastImageData = null;
    this.sampler = null;
  }
};
export {
  WebGpuDetector
};
