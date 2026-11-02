/** VLFeat SIFT runs in WASM. No scene recognition or stored image coordinates. */
export function siftSeeds(s: any, cv: any, ctx: any, log = (_: string) => {}) {
 function features(g: ArrayLike<number>, w: number, h: number) {
  const p = s._malloc(w*h*4);
  if (!p) throw Error('SIFT allocation failed');
  try { s.HEAPF32.set(g,p/4); const n=s._extract(p,w,h); return {n,a:s.HEAPF32.slice(s._get_out()/4,s._get_out()/4+n*130)}; }
  finally { s._free(p); }
 }
 const S=256, ref=Float32Array.from({length:S*S},(_,i)=>ctx.ref[Math.floor(i/S/16)][Math.floor(i%S/16)]);
 const a=features(ref,S,S), b=features(ctx.g,ctx.w,ctx.h);
 log(`SIFT WASM: ${a.n} texture / ${b.n} image features`);
 let P:number[][]=[], Q:number[][]=[];
 for(let i=0;i<b.n;i++) {
  let best=Infinity,second=Infinity,jbest=0;
  for(let j=0;j<a.n;j++) { let d=0; for(let k=2;k<130;k++){const v=b.a[i*130+k]-a.a[j*130+k];d+=v*v;}
   if(d<best){second=best;best=d;jbest=j;}else if(d<second)second=d;
  }
  if(best<.68*.68*second){P.push([a.a[jbest*130],a.a[jbest*130+1]]);Q.push([b.a[i*130],b.a[i*130+1]]);}
 }
 log(`SIFT: ${P.length} ratio-test matches`);
 const found:number[][][]=[];
 for(let it=0;it<20&&P.length>=10;it++) {
  const sp=cv.matFromArray(P.length,1,cv.CV_32FC2,P.flat()), dp=cv.matFromArray(Q.length,1,cv.CV_32FC2,Q.flat()), mask=new cv.Mat();
  let H:any;
  try {
   H=cv.findHomography(sp,dp,cv.RANSAC,2.5,mask,2500,.99);
   if(!H||H.empty()||Array.from(mask.data).filter(Boolean).length<8)break;
   const m=Array.from(H.data64F) as number[];
   P=P.filter((_,j)=>!mask.data[j]);Q=Q.filter((_,j)=>!mask.data[j]);
   const q=[[0,0],[S,0],[S,S],[0,S]].map(([x,y])=>{const d=m[6]*x+m[7]*y+m[8];return [(m[0]*x+m[1]*y+m[2])/d,(m[3]*x+m[4]*y+m[5])/d];});
   if(q.some(p=>!p.every(Number.isFinite)||p[0]<0||p[1]<0||p[0]>=ctx.w-1||p[1]>=ctx.h-1))continue;
   let sign=0, convex=true;
   for(let k=0;k<4;k++){const a=q[k],b=q[(k+1)%4],c=q[(k+2)%4],z=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);if(!z||(sign&&Math.sign(z)!==sign))convex=false;sign=Math.sign(z);}
   const area=Math.abs(q.reduce((v,p,k)=>v+p[0]*q[(k+1)%4][1]-q[(k+1)%4][0]*p[1],0))/2;
   if(!convex||area<1000||area>ctx.w*ctx.h*.2)continue;
   let start=0;q.forEach((p,k)=>{if(p[0]+p[1]<q[start][0]+q[start][1])start=k;});let canon=[0,1,2,3].map(k=>q[(k+start)%4]);if(sign<0)canon=[canon[0],canon[3],canon[2],canon[1]];
   found.push(canon);
  } finally {sp.delete();dp.delete();mask.delete();if(H)H.delete();}
 }
 log(`SIFT: ${found.length} bounded tile hypotheses`);
 return found;
}
