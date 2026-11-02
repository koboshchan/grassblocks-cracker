// u64 represented as vec2<u32>(low,high). All arithmetic wraps mod 2^64.
fn hi32(a:u32,b:u32)->u32 {
 let al=a&65535u;let ah=a>>16u;let bl=b&65535u;let bh=b>>16u;
 let p0=al*bl;let mid=al*bh+(p0>>16u);let mid2=(mid&65535u)+ah*bl;
 return ah*bh+(mid>>16u)+(mid2>>16u);
}
fn mulc(a:vec2<u32>,b:u32)->vec2<u32>{return vec2(a.x*b,hi32(a.x,b)+a.y*b);}
fn add64(a:vec2<u32>,b:vec2<u32>)->vec2<u32>{let l=a.x+b.x;return vec2(l,a.y+b.y+select(0u,1u,l<a.x));}
fn rotation(x:i32,y:i32,z:i32)->u32{
 let xp=x*3129871;
 let zp=mulc(vec2(u32(z),select(0u,0xffffffffu,z<0)),116129781u);
 let l=vec2(u32(xp)^zp.x^u32(y),select(0u,0xffffffffu,xp<0)^zp.y^select(0u,0xffffffffu,y<0));
 let sq=vec2(l.x*l.x,hi32(l.x,l.x)+2u*l.x*l.y);
 let p=add64(mulc(sq,42317861u),mulc(l,11u));
 let s=vec2(((p.x>>16u)|(p.y<<16u))^0xdeece66du,(p.y>>16u)^5u);
 let low=s.x*0xdeece66du;
 let high=hi32(s.x,0xdeece66du)+s.y*0xdeece66du+s.x*5u+select(0u,1u,low>0xfffffff4u);
 return (high>>14u)&3u;
}
struct Block {x:i32,y:i32,z:i32,r:i32,side:i32}
struct Params {xmin:i32,zmin:i32,ymin:i32,ymax:i32,width:u32,offset:u32,count:u32,blocks:u32}
struct Results {count:atomic<u32>,xyz:array<vec4<i32>>}
@group(0) @binding(0) var<storage,read> blocks:array<Block>;
@group(0) @binding(1) var<uniform> p:Params;
@group(0) @binding(2) var<storage,read_write> results:Results;
@compute @workgroup_size(256)
fn search(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x>=p.count){return;}
 let pos=p.offset+id.x;
 let x=p.xmin+i32(pos%p.width);let z=p.zmin+i32(pos/p.width);
 for(var y=p.ymin;y<=p.ymax;y++){
  var match=true;
  for(var i=0u;i<p.blocks;i++){
   let b=blocks[i];let r=rotation(x+b.x,y+b.y,z+b.z);
   let mask=select(3u,1u,b.side!=0);
   if((r&mask)!=u32(b.r)){match=false;break;}
  }
  if(match){let i=atomicAdd(&results.count,1u);if(i<4096u){results.xyz[i]=vec4(x,y,z,0);}}
 }
}
