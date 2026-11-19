// u64 represented as vec2<u32>(low,high). All arithmetic wraps mod 2^64.
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
fn hi32_116129781(a:u32)->u32 {
 let al=a&65535u;let ah=a>>16u;
 let p0=al*65525u;let mid=al*1771u+(p0>>16u);let mid2=(mid&65535u)+ah*65525u;
 return ah*1771u+(mid>>16u)+(mid2>>16u);
}
fn mulc_z(z:i32)->vec2<u32>{
 let zu=u32(z);let zh=u32(z>>31u);
 return vec2(zu*116129781u,hi32_116129781(zu)+zh*116129781u);
}
fn rotation_xz_zp(x:i32,zp:vec2<u32>)->vec2<u32>{
 let xp=x*3129871;
 return vec2(u32(xp)^zp.x,u32(xp>>31u)^zp.y);
}
fn rotation_xz(x:i32,z:i32)->vec2<u32>{
 let xp=x*3129871;
 let zp=mulc_z(z);
 return vec2(u32(xp)^zp.x,u32(xp>>31u)^zp.y);
}
fn rotation_y_lh(xz_lo:u32,k1:u32,k0:u32,y_u:u32)->u32{
 let lx=xz_lo^y_u;
 let sqx=lx*lx;
 let px=sqx*42317861u;let ll=lx*11u;
 let rx=px+ll;
 let ry=hi32_42317861(sqx)+hi32_sq(lx)*42317861u+lx*k1+(((lx>>16u)*11u+(((lx&65535u)*11u)>>16u))>>16u)+k0+select(0u,1u,rx<px);
 let sx=((rx>>16u)|(ry<<16u))^0xdeece66du;
 let sy=(ry>>16u)^5u;
 let low=sx*0xdeece66du;
 let high=hi32_deec(sx)+sy*0xdeece66du+sx*5u+select(0u,1u,low>0xfffffff4u);
 return (high>>14u)&3u;
}
fn rotation_y(xz:vec2<u32>,y:i32)->u32{
 let lx=xz.x^u32(y);let ly=xz.y^u32(y>>31u);
 let sqx=lx*lx;let sqy=hi32_sq(lx)+2u*lx*ly;
 let px=sqx*42317861u;let ll=lx*11u;
 let rx=px+ll;
 let ry=hi32_42317861(sqx)+sqy*42317861u+(((lx>>16u)*11u+(((lx&65535u)*11u)>>16u))>>16u)+ly*11u+select(0u,1u,rx<px);
 let sx=((rx>>16u)|(ry<<16u))^0xdeece66du;
 let sy=(ry>>16u)^5u;
 let low=sx*0xdeece66du;
 let high=hi32_deec(sx)+sy*0xdeece66du+sx*5u+select(0u,1u,low>0xfffffff4u);
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
override SINGLE_NORTH:i32=-1;
override ALL_FOUR:bool=false;

fn check_rest(x:i32,y:i32,z:i32,n:u32,xz1:vec2<u32>,b1:Block,mask1:u32,b2:Block,mask2:u32,b3:Block,mask3:u32)->bool{
 if(p.blocks>1u){
  let r1=rotation_y(xz1,y+b1.y);
  if((r1&mask1)!=u32(b1.r[n])){return false;}
 }
 if(p.blocks>2u){
  let r2=rotation(x+b2.x[n],y+b2.y,z+b2.z[n]);
  if((r2&mask2)!=u32(b2.r[n])){return false;}
 }
 if(p.blocks>3u){
  let r3=rotation(x+b3.x[n],y+b3.y,z+b3.z[n]);
  if((r3&mask3)!=u32(b3.r[n])){return false;}
 }
 for(var i=4u;i<p.blocks;i++){
  let b=blocks[i];let r=rotation(x+b.x[n],y+b.y,z+b.z[n]);
  let mask=select(3u,1u,b.side!=0);
  if((r&mask)!=u32(b.r[n])){return false;}
 }
 return true;
}

@compute @workgroup_size(WG_SIZE)
fn search(@builtin(global_invocation_id) id:vec3<u32>){
 let base_index=id.x*ITEMS;
 if(base_index>=p.count||p.blocks==0u){return;}
 let base_pos=p.offset+base_index;
 var qz=base_pos/p.width;
 var rx=base_pos-qz*p.width;
 let b0=blocks[0];
 let anchor_r=u32(select(select(select(b0.r.w-3,b0.r.z-2,b0.r.z>=0),b0.r.y-1,b0.r.y>=0),b0.r.x,b0.r.x>=0))&3u;
 var b0_mask=0u;
 if(!ALL_FOUR&&SINGLE_NORTH<0){
  if(b0.r.x>=0){b0_mask|=1u;}
  if(b0.r.y>=0){b0_mask|=2u;}
  if(b0.r.z>=0){b0_mask|=4u;}
  if(b0.r.w>=0){b0_mask|=8u;}
 }
 var b1:Block;var mask1=0u;
 if(p.blocks>1u){b1=blocks[1];mask1=select(3u,1u,b1.side!=0);}
 var b2:Block;var mask2=0u;
 if(p.blocks>2u){b2=blocks[2];mask2=select(3u,1u,b2.side!=0);}
 var b3:Block;var mask3=0u;
 if(p.blocks>3u){b3=blocks[3];mask3=select(3u,1u,b3.side!=0);}
 var zp0=mulc_z(p.zmin+i32(qz));
 for(var item=0u;item<ITEMS;item++){
  let index=base_index+item;
  if(index>=p.count){return;}
  let x=p.xmin+i32(rx);let z=p.zmin+i32(qz);
  let xz0=rotation_xz_zp(x,zp0);
  if(SINGLE_NORTH>=0){
   let sn=u32(SINGLE_NORTH);
   var xz1=vec2<u32>(0u);
   if(p.blocks>1u){xz1=rotation_xz(x+b1.x[sn],z+b1.z[sn]);}
   let target0=u32(b0.r[sn]);
   let is_side=b0.side!=0;
   for(var seg=0u;seg<2u;seg++){
    var ya=p.ymin;var yb=p.ymax;
    if(seg==0u){yb=min(p.ymax,-b0.y-1);}else{ya=max(p.ymin,-b0.y);}
    if(ya>yb){continue;}
    let lh=select(xz0.y,~xz0.y,seg==0u);
    let k1=lh*84635722u;let k0=lh*11u;
    for(var y=ya;y<=yb;y++){
     let r0=rotation_y_lh(xz0.x,k1,k0,u32(y+b0.y));
     let hit=select(r0==target0,(r0&1u)==target0,is_side);
     if(hit&&check_rest(x,y,z,sn,xz1,b1,mask1,b2,mask2,b3,mask3)){
      let idx=atomicAdd(&results.count,1u);
      if(idx<4096u){results.xyz[idx]=vec4(x,y,z,i32(sn));}
     }
    }
   }
  }else{
   var xz1_lo=vec4<u32>(0u);var xz1_hi=vec4<u32>(0u);
   if(p.blocks>1u){
    if(ALL_FOUR||b0.r.x>=0){let t=rotation_xz(x+b1.x.x,z+b1.z.x);xz1_lo.x=t.x;xz1_hi.x=t.y;}
    if(ALL_FOUR||b0.r.y>=0){let t=rotation_xz(x+b1.x.y,z+b1.z.y);xz1_lo.y=t.x;xz1_hi.y=t.y;}
    if(ALL_FOUR||b0.r.z>=0){let t=rotation_xz(x+b1.x.z,z+b1.z.z);xz1_lo.z=t.x;xz1_hi.z=t.y;}
    if(ALL_FOUR||b0.r.w>=0){let t=rotation_xz(x+b1.x.w,z+b1.z.w);xz1_lo.w=t.x;xz1_hi.w=t.y;}
   }
   let side_r=anchor_r&1u;
   for(var seg=0u;seg<2u;seg++){
    var ya=p.ymin;var yb=p.ymax;
    if(seg==0u){yb=min(p.ymax,-b0.y-1);}else{ya=max(p.ymin,-b0.y);}
    if(ya>yb){continue;}
    let lh=select(xz0.y,~xz0.y,seg==0u);
    let k1=lh*84635722u;let k0=lh*11u;
    if(b0.side==0){
     for(var y=ya;y<=yb;y++){
      let n=(rotation_y_lh(xz0.x,k1,k0,u32(y+b0.y))-anchor_r)&3u;
      if(!ALL_FOUR&&((b0_mask>>n)&1u)==0u){continue;}
      if(check_rest(x,y,z,n,vec2(xz1_lo[n],xz1_hi[n]),b1,mask1,b2,mask2,b3,mask3)){
       let idx=atomicAdd(&results.count,1u);
       if(idx<4096u){results.xyz[idx]=vec4(x,y,z,i32(n));}
      }
     }
    }else{
     for(var y=ya;y<=yb;y++){
      let p0=(rotation_y_lh(xz0.x,k1,k0,u32(y+b0.y))-side_r)&1u;
      for(var s=0u;s<2u;s++){
       let n=p0+(s<<1u);
       if(!ALL_FOUR&&((b0_mask>>n)&1u)==0u){continue;}
       if(check_rest(x,y,z,n,vec2(xz1_lo[n],xz1_hi[n]),b1,mask1,b2,mask2,b3,mask3)){
        let idx=atomicAdd(&results.count,1u);
        if(idx<4096u){results.xyz[idx]=vec4(x,y,z,i32(n));}
       }
      }
     }
    }
   }
  }
  rx++;
  if(rx==p.width){
   rx=0u;
   qz++;
   zp0=mulc_z(p.zmin+i32(qz));
  }
 }
}
