// u64 represented as vec2<u32>(low,high). All arithmetic wraps mod 2^64.
fn hi32(a:u32,b:u32)->u32 {
 let al=a&65535u;let ah=a>>16u;let bl=b&65535u;let bh=b>>16u;
 let p0=al*bl;let mid=al*bh+(p0>>16u);let mid2=(mid&65535u)+ah*bl;
 return ah*bh+(mid>>16u)+(mid2>>16u);
}
fn hi32_sq(a:u32)->u32 {
 let al=a&65535u;let ah=a>>16u;
 let p0=al*al;let prod=al*ah;
 let mid=prod+(p0>>16u);let mid2=(mid&65535u)+prod;
 return ah*ah+(mid>>16u)+(mid2>>16u);
}
fn hi32_deec(a:u32)->u32 {
 let al=a&65535u;let ah=a>>16u;
 let p0=al*0xe66du;let mid=al*0xdeecu+(p0>>16u);let mid2=(mid&65535u)+ah*0xe66du;
 return ah*0xdeecu+(mid>>16u)+(mid2>>16u);
}
fn hi32_42317861(a:u32)->u32 {
 let al=a&65535u;let ah=a>>16u;
 let p0=al*47141u;let mid=al*645u+(p0>>16u);let mid2=(mid&65535u)+ah*47141u;
 return ah*645u+(mid>>16u)+(mid2>>16u);
}
fn mulc(a:vec2<u32>,b:u32)->vec2<u32>{return vec2(a.x*b,hi32(a.x,b)+a.y*b);}
fn mulc_42317861(a:vec2<u32>)->vec2<u32>{return vec2(a.x*42317861u,hi32_42317861(a.x)+a.y*42317861u);}
fn mul11(a:vec2<u32>)->vec2<u32>{
 return vec2(a.x*11u,(((a.x>>16u)*11u+(((a.x&65535u)*11u)>>16u))>>16u)+a.y*11u);
}
fn add64(a:vec2<u32>,b:vec2<u32>)->vec2<u32>{let l=a.x+b.x;return vec2(l,a.y+b.y+select(0u,1u,l<a.x));}
fn rotation_xz(x:i32,z:i32)->vec2<u32>{
 let xp=x*3129871;
 let zp=mulc(vec2(u32(z),select(0u,0xffffffffu,z<0)),116129781u);
 return vec2(u32(xp)^zp.x,select(0u,0xffffffffu,xp<0)^zp.y);
}
fn rotation_y(xz:vec2<u32>,y:i32)->u32{
 let l=vec2(xz.x^u32(y),xz.y^select(0u,0xffffffffu,y<0));
 let sq=vec2(l.x*l.x,hi32_sq(l.x)+2u*l.x*l.y);
 let p=add64(mulc_42317861(sq),mul11(l));
 let s=vec2(((p.x>>16u)|(p.y<<16u))^0xdeece66du,(p.y>>16u)^5u);
 let low=s.x*0xdeece66du;
 let high=hi32_deec(s.x)+s.y*0xdeece66du+s.x*5u+select(0u,1u,low>0xfffffff4u);
 return (high>>14u)&3u;
}
fn rotation(x:i32,y:i32,z:i32)->u32{
 return rotation_y(rotation_xz(x,z),y);
}
struct Block {x:vec4<i32>,z:vec4<i32>,r:vec4<i32>,y:i32,side:i32}
struct Params {xmin:i32,zmin:i32,ymin:i32,ymax:i32,width:u32,offset:u32,count:u32,blocks:u32}
struct Results {count:atomic<u32>,xyz:array<vec4<i32>>}
@group(0) @binding(0) var<storage,read> blocks:array<Block>;
@group(0) @binding(1) var<uniform> p:Params;
@group(0) @binding(2) var<storage,read_write> results:Results;
override WG_SIZE:u32=256u;
override ITEMS:u32=1u;
@compute @workgroup_size(WG_SIZE)
fn search(@builtin(global_invocation_id) id:vec3<u32>){
 let base_index=id.x*ITEMS;
 if(base_index>=p.count||p.blocks==0u){return;}
 let base_pos=p.offset+base_index;
 var qz=base_pos/p.width;
 var rx=base_pos-qz*p.width;
 let b0=blocks[0];
 let target0=u32(b0.r.x)&3u;
 var b1:Block;
 var mask1=0u;
 if(p.blocks>1u){
  b1=blocks[1];
  mask1=select(3u,1u,b1.side!=0);
 }
 var b2:Block;
 var mask2=0u;
 if(p.blocks>2u){
  b2=blocks[2];
  mask2=select(3u,1u,b2.side!=0);
 }
 var b3:Block;
 var mask3=0u;
 if(p.blocks>3u){
  b3=blocks[3];
  mask3=select(3u,1u,b3.side!=0);
 }
 for(var item=0u;item<ITEMS;item++){
  let index=base_index+item;
  if(index>=p.count){return;}
  let x=p.xmin+i32(rx);let z=p.zmin+i32(qz);
  let xz0=rotation_xz(x,z);
  for(var y=p.ymin;y<=p.ymax;y++){
   let r0_val=rotation_y(xz0,y+b0.y);
   let n=(r0_val-target0)&3u;
   if(b0.r[n]<0||(b0.side!=0&&(u32(b0.side)&(1u<<n))==0u)){continue;}
   if(p.blocks>1u){
    let r1=rotation(x+b1.x[n],y+b1.y,z+b1.z[n]);
    if((r1&mask1)!=u32(b1.r[n])){continue;}
   }
   if(p.blocks>2u){
    let r2=rotation(x+b2.x[n],y+b2.y,z+b2.z[n]);
    if((r2&mask2)!=u32(b2.r[n])){continue;}
   }
   if(p.blocks>3u){
    let r3=rotation(x+b3.x[n],y+b3.y,z+b3.z[n]);
    if((r3&mask3)!=u32(b3.r[n])){continue;}
   }
   var matched=true;
   for(var i=4u;i<p.blocks;i++){
    let b=blocks[i];let r=rotation(x+b.x[n],y+b.y,z+b.z[n]);
    let mask=select(3u,1u,b.side!=0);
    if((r&mask)!=u32(b.r[n])){matched=false;break;}
   }
   if(matched){let idx=atomicAdd(&results.count,1u);if(idx<4096u){results.xyz[idx]=vec4(x,y,z,i32(n));}}
  }
  rx++;
  if(rx==p.width){rx=0u;qz++;}
 }
}
