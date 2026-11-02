/* Freestanding browser detector. SIMD lanes are rotations, preserving JS summation order. */
#include <wasm_simd128.h>
typedef unsigned char u8;
static double refs[256][4];
static float patch[256];
void set_refs(const u8 *r){for(int i=0;i<256;i++)for(int k=0;k<4;k++)refs[i][k]=r[k*256+i];}
static int fit(const float*q,double*H){
 double m[8][9]={0};
 for(int i=0;i<4;i++){
  double x=(i==1||i==2)?16:0,y=i>=2?16:0,u=q[i*2],v=q[i*2+1];
  double rows[2][9]={{x,y,1,0,0,0,-u*x,-u*y,u},{0,0,0,x,y,1,-v*x,-v*y,v}};
  for(int k=0;k<2;k++)for(int a=0;a<8;a++){m[a][8]+=rows[k][a]*rows[k][8];for(int b=0;b<8;b++)m[a][b]+=rows[k][a]*rows[k][b];}
 }
 for(int j=0;j<8;j++){
  int k=j;for(int i=j+1;i<8;i++)if(__builtin_fabs(m[i][j])>__builtin_fabs(m[k][j]))k=i;
  for(int t=0;t<9;t++){double tmp=m[j][t];m[j][t]=m[k][t];m[k][t]=tmp;}
  if(__builtin_fabs(m[j][j])<1e-12)return 0;
  double d=m[j][j];for(int t=j;t<=8;t++)m[j][t]/=d;
  for(int i=0;i<8;i++)if(i!=j){double f=m[i][j];if(f)for(int t=j;t<=8;t++)m[i][t]-=f*m[j][t];}
 }
 for(int i=0;i<8;i++)H[i]=m[i][8];H[8]=1;return 1;
}
static int correlate(int quadrant,double *sc){
 double sa=0,aa=0;int n=0;
 v128_t sb0=wasm_f64x2_splat(0),sb1=sb0,bb0=sb0,bb1=sb0,ab0=sb0,ab1=sb0;
 for(int j=1;j<15;j++)for(int i=1;i<15;i++){
  if(quadrant>=0&&((i>=8?1:0)+(j>=8?2:0))!=quadrant)continue;
  int idx=j*16+i;double a=patch[idx];sa+=a;aa+=a*a;n++;
  v128_t av=wasm_f64x2_splat(a),b0=wasm_v128_load(refs[idx]),b1=wasm_v128_load(refs[idx]+2);
  sb0=wasm_f64x2_add(sb0,b0);sb1=wasm_f64x2_add(sb1,b1);
  bb0=wasm_f64x2_add(bb0,wasm_f64x2_mul(b0,b0));bb1=wasm_f64x2_add(bb1,wasm_f64x2_mul(b1,b1));
  ab0=wasm_f64x2_add(ab0,wasm_f64x2_mul(av,b0));ab1=wasm_f64x2_add(ab1,wasm_f64x2_mul(av,b1));
 }
 double sb[4],bb[4],ab[4];wasm_v128_store(sb,sb0);wasm_v128_store(sb+2,sb1);wasm_v128_store(bb,bb0);wasm_v128_store(bb+2,bb1);wasm_v128_store(ab,ab0);wasm_v128_store(ab+2,ab1);
 int best=0;double va=aa-sa*sa/n;
 for(int k=0;k<4;k++){double vb=bb[k]-sb[k]*sb[k]/n;sc[k]=(va<1e-9||vb<1e-9)?0:(ab[k]-sa*sb[k]/n)/__builtin_sqrt(va*vb);if(sc[k]>sc[best])best=k;}
 return best;
}
void classify_batch(const u8*g,int w,int h,const float*quads,int count,float*out){
 for(int t=0;t<count;t++){
  float*o=out+t*5;for(int k=0;k<5;k++)o[k]=0;
  double H[9];if(!fit(quads+t*8,H))continue;int valid=1;
  for(int j=0;j<16&&valid;j++)for(int i=0;i<16;i++){
   double x=i+.5,y=j+.5,d=H[6]*x+H[7]*y+H[8],px=(H[0]*x+H[1]*y+H[2])/d,py=(H[3]*x+H[4]*y+H[5])/d;
   if(!(px>=0&&py>=0&&px<w-1&&py<h-1)){valid=0;break;}
   int xx=(int)px,yy=(int)py;double fx=px-xx,fy=py-yy;int p=yy*w+xx;
   patch[j*16+i]=(float)(g[p]*(1-fx)*(1-fy)+g[p+1]*fx*(1-fy)+g[p+w]*(1-fx)*fy+g[p+w+1]*fx*fy);
  }
  if(!valid)continue;
  double sc[4];int best=correlate(-1,sc),second=best==0?1:0;
  for(int k=0;k<4;k++)if(k!=best&&sc[k]>sc[second])second=k;
  int votes=0;for(int q=0;q<4;q++){double s[4];if(correlate(q,s)==best)votes++;}
  o[0]=1;o[1]=best;o[2]=(float)sc[best];o[3]=(float)(sc[best]-sc[second]);o[4]=votes;
 }
}

void *memset(void*p,int c,unsigned long n){unsigned char*q=p;for(unsigned long i=0;i<n;i++)q[i]=(unsigned char)c;return p;}
