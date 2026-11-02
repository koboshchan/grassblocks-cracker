#include <stdint.h>
#include <wasm_simd128.h>
typedef struct {int32_t x,y,z,r,side;} Block;
static Block blocks[512];static int32_t results[32768];
int blocks_ptr(void){return (int)(uintptr_t)blocks;}int results_ptr(void){return (int)(uintptr_t)results;}
static uint32_t rot(int32_t x,int32_t y,int32_t z){
 uint64_t l=(uint64_t)(int64_t)(int32_t)((uint32_t)x*3129871u) ^ ((uint64_t)(int64_t)z*116129781ull) ^ (uint64_t)(int64_t)y;
 l=l*l*42317861ull+l*11ull;
 // Logical shift is enough here: only low 48 bits matter after the Java LCG.
 uint64_t s=(l>>16)^0x5deece66dull;
 return (uint32_t)(((s*0x5deece66dull+11ull)>>46)&3);
}
typedef uint32_t V __attribute__((vector_size(16)));
static V high(V a,V b){V al=a&65535,ah=a>>16,bl=b&65535,bh=b>>16;V p0=al*bl,mid=al*bh+(p0>>16),mid2=(mid&65535)+ah*bl;return ah*bh+(mid>>16)+(mid2>>16);}
static V vr(V x,V y,V z){
 V zero={0,0,0,0},c={116129781,116129781,116129781,116129781};
 V xp=x*3129871, zl=z*c,zh=high(z,c)+((V)((__attribute__((vector_size(16))) int32_t)z<(__attribute__((vector_size(16))) int32_t)zero)&(zero-c));
 V lo=xp^zl^y,hi=(V)((__attribute__((vector_size(16))) int32_t)xp<(__attribute__((vector_size(16))) int32_t)zero)^zh^(V)((__attribute__((vector_size(16))) int32_t)y<(__attribute__((vector_size(16))) int32_t)zero);
 V sq=lo*lo,sqh=high(lo,lo)+2*lo*hi,cc={42317861,42317861,42317861,42317861};
 V pl=sq*cc,ph=high(sq,cc)+sqh*cc,eleven={11,11,11,11};
 V ll=lo*11,lh=high(lo,eleven)+hi*11,rl=pl+ll,rh=ph+lh+(((V)(rl<pl))&1);
 V sl=((rl>>16)|(rh<<16))^0xdeece66d,sh=(rh>>16)^5,lc={0xdeece66d,0xdeece66d,0xdeece66d,0xdeece66d};
 V low=sl*lc,hh=high(sl,lc)+sh*lc+sl*5+(((V)(low>(zero-12)))&1);
 return (hh>>14)&3;
}
int rotation(int x,int y,int z){return rot(x,y,z);}
int scan(int32_t x,int32_t y,int32_t z,uint32_t count,uint32_t n){
 uint32_t out=0;
 for(uint32_t i=0;i<count;i+=4){
 Block b=blocks[0]; V xx={(uint32_t)x+i+b.x,(uint32_t)x+i+1+b.x,(uint32_t)x+i+2+b.x,(uint32_t)x+i+3+b.x}, yy={(uint32_t)y+b.y,(uint32_t)y+b.y,(uint32_t)y+b.y,(uint32_t)y+b.y},zz={(uint32_t)z+b.z,(uint32_t)z+b.z,(uint32_t)z+b.z,(uint32_t)z+b.z};
 V r=vr(xx,yy,zz);
 for(uint32_t lane=0;lane<4&&i+lane<count;lane++){
 if((r[lane]&(b.side?1:3))!=(uint32_t)b.r)continue;
 int32_t ax=(int32_t)((uint32_t)x+i+lane);int ok=1;
 for(uint32_t j=1;j<n;j++){Block q=blocks[j];if((rot(ax+q.x,y+q.y,z+q.z)&(q.side?1:3))!=(uint32_t)q.r){ok=0;break;}}
 if(ok)results[out++]=ax;
 }}
 return out;
}
