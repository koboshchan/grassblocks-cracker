/* Freestanding browser detector. SIMD lanes are rotations, preserving JS summation order. */
#include <wasm_simd128.h>
typedef unsigned char u8;
static double refs[256][4];
static float patch[256];
static double sb[4], bb[4], vb[4];
static double q_sb[4][4], q_bb[4][4], q_vb[4][4];
void set_refs(const u8 *r){
 for(int i=0;i<256;i++)for(int k=0;k<4;k++)refs[i][k]=r[k*256+i];
 for(int k=0;k<4;k++){sb[k]=0;bb[k]=0;}
 for(int q=0;q<4;q++)for(int k=0;k<4;k++){q_sb[q][k]=0;q_bb[q][k]=0;}
 for(int j=1;j<15;j++)for(int i=1;i<15;i++){
  int q=(i>=8?1:0)+(j>=8?2:0);
  int idx=j*16+i;
  for(int k=0;k<4;k++){
   double b=refs[idx][k];
   sb[k]+=b;bb[k]+=b*b;
   q_sb[q][k]+=b;q_bb[q][k]+=b*b;
  }
 }
 for(int k=0;k<4;k++)vb[k]=bb[k]-sb[k]*sb[k]/196.0;
 for(int q=0;q<4;q++)for(int k=0;k<4;k++)q_vb[q][k]=q_bb[q][k]-q_sb[q][k]*q_sb[q][k]/49.0;
}
static int fit(const float*q,double*H){
 double x0=q[0],y0=q[1],x1=q[2],y1=q[3],x2=q[4],y2=q[5],x3=q[6],y3=q[7];
 double dx1=x1-x2,dx2=x3-x2,dx3=x0-x1+x2-x3;
 double dy1=y1-y2,dy2=y3-y2,dy3=y0-y1+y2-y3;
 double det=dx1*dy2-dx2*dy1;
 if(__builtin_fabs(det)<1e-12)return 0;
 double g=((dx3*dy2-dx2*dy3)/det)/16.0;
 double h=((dx1*dy3-dx3*dy1)/det)/16.0;
 double a=(x1-x0)/16.0+g*x1;
 double b=(x3-x0)/16.0+h*x3;
 double c=x0;
 double d=(y1-y0)/16.0+g*y1;
 double e=(y3-y0)/16.0+h*y3;
 double f=y0;
 H[0]=a;H[1]=b;H[2]=c;H[3]=d;H[4]=e;H[5]=f;H[6]=g;H[7]=h;H[8]=1.0;
 return 1;
}
void classify_batch(const u8*g,int w,int h,const float*quads,int count,float*out){
 for(int t=0;t<count;t++){
  float*o=out+t*5;for(int k=0;k<5;k++)o[k]=0;
  double H[9];if(!fit(quads+t*8,H))continue;int valid=1;
  for(int j=1;j<15&&valid;j++){
   double num_x=H[0]*1.5+H[1]*(j+0.5)+H[2];
   double num_y=H[3]*1.5+H[4]*(j+0.5)+H[5];
   double den=H[6]*1.5+H[7]*(j+0.5)+H[8];
   for(int i=1;i<15;i++){
    double px=num_x/den,py=num_y/den;
    if(!(px>=0&&py>=0&&px<w-1&&py<h-1)){valid=0;break;}
    int xx=(int)px,yy=(int)py;double fx=px-xx,fy=py-yy;int p=yy*w+xx;
    patch[j*16+i]=(float)(g[p]*(1-fx)*(1-fy)+g[p+1]*fx*(1-fy)+g[p+w]*(1-fx)*fy+g[p+w+1]*fx*fy);
    num_x+=H[0];num_y+=H[3];den+=H[6];
   }
  }
  if(!valid)continue;
  double q_sa[4]={0},q_aa[4]={0};
  v128_t q_ab0[4],q_ab1[4];
  v128_t zero=wasm_f64x2_splat(0);
  for(int q=0;q<4;q++){
   q_ab0[q]=zero;
   q_ab1[q]=zero;
  }
  for(int j=1;j<15;j++)for(int i=1;i<15;i++){
   int q=(i>=8?1:0)+(j>=8?2:0);
   int idx=j*16+i;double a=patch[idx];
   q_sa[q]+=a;q_aa[q]+=a*a;
   v128_t av=wasm_f64x2_splat(a),b0=wasm_v128_load(refs[idx]),b1=wasm_v128_load(refs[idx]+2);
   q_ab0[q]=wasm_f64x2_add(q_ab0[q],wasm_f64x2_mul(av,b0));
   q_ab1[q]=wasm_f64x2_add(q_ab1[q],wasm_f64x2_mul(av,b1));
  }
  double sa=q_sa[0]+q_sa[1]+q_sa[2]+q_sa[3];
  double aa=q_aa[0]+q_aa[1]+q_aa[2]+q_aa[3];
  v128_t ab0=wasm_f64x2_add(wasm_f64x2_add(wasm_f64x2_add(q_ab0[0],q_ab0[1]),q_ab0[2]),q_ab0[3]);
  v128_t ab1=wasm_f64x2_add(wasm_f64x2_add(wasm_f64x2_add(q_ab1[0],q_ab1[1]),q_ab1[2]),q_ab1[3]);
  double ab[4];
  wasm_v128_store(ab,ab0);wasm_v128_store(ab+2,ab1);
  int best=0;double va=aa-sa*sa/196.0;double sc[4];
  for(int k=0;k<4;k++){
   sc[k]=(va<1e-9||vb[k]<1e-9)?0:(ab[k]-sa*sb[k]/196.0)/__builtin_sqrt(va*vb[k]);
   if(sc[k]>sc[best])best=k;
  }
  int second=best==0?1:0;
  for(int k=0;k<4;k++)if(k!=best&&sc[k]>sc[second])second=k;
  int votes=0;
  for(int q=0;q<4;q++){
   double qab[4];
   wasm_v128_store(qab,q_ab0[q]);wasm_v128_store(qab+2,q_ab1[q]);
   double qva=q_aa[q]-q_sa[q]*q_sa[q]/49.0;
   int q_best=0;double q_sc[4];
   for(int k=0;k<4;k++){
    q_sc[k]=(qva<1e-9||q_vb[q][k]<1e-9)?0:(qab[k]-q_sa[q]*q_sb[q][k]/49.0)/__builtin_sqrt(qva*q_vb[q][k]);
    if(q_sc[k]>q_sc[q_best])q_best=k;
   }
   if(q_best==best)votes++;
  }
  o[0]=1;o[1]=best;o[2]=(float)sc[best];o[3]=(float)(sc[best]-sc[second]);o[4]=votes;
 }
}
void *memset(void*p,int c,unsigned long n){unsigned char*q=p;for(unsigned long i=0;i<n;i++)q[i]=(unsigned char)c;return p;}
