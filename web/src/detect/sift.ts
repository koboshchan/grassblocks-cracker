/** VLFeat SIFT runs in WASM. No scene recognition or stored image coordinates. */
// First octave doubles each dimension. Bound its pyramid, not just the input buffer.
export const SIFT_MAX_PIXELS = 1024 * 1024;
export function prepareSiftImage(g: ArrayLike<number>, w: number, h: number) {
 if (!Number.isSafeInteger(w) || !Number.isSafeInteger(h) || w < 8 || h < 8 || w * h > 16 * 1024 * 1024 || g.length !== w * h)
  throw Error('Invalid SIFT grayscale dimensions or stride');
 const scale = Math.min(1, Math.sqrt(SIFT_MAX_PIXELS / (w * h)), 2048 / Math.max(w,h));
 const width = Math.max(8, Math.floor(w * scale)), height = Math.max(8, Math.floor(h * scale));
 const sx = w / width, sy = h / height;
 if (width === w && height === h) return {data:g,width,height,sx,sy};
 const data = new Float32Array(width * height);
 for (let y=0;y<height;y++) {
  const fy=(y+.5)*sy-.5, y0=Math.max(0,Math.floor(fy)), y1=Math.min(h-1,y0+1), dy=fy-y0;
  for (let x=0;x<width;x++) {
   const fx=(x+.5)*sx-.5, x0=Math.max(0,Math.floor(fx)), x1=Math.min(w-1,x0+1), dx=fx-x0;
   data[y*width+x]=(1-dy)*((1-dx)*g[y0*w+x0]+dx*g[y0*w+x1])+dy*((1-dx)*g[y1*w+x0]+dx*g[y1*w+x1]);
  }
 }
 return {data,width,height,sx,sy};
}
export function extractSiftFeatures(s:any, g:ArrayLike<number>, w:number, h:number) {
 const image=prepareSiftImage(g,w,h), bytes=image.width*image.height*4;
 const p=s._malloc(bytes);
 if (!p || p % 4 || p + bytes > s.HEAPF32.byteLength) throw Error('SIFT allocation failed');
 try {
  s.HEAPF32.set(image.data,p/4);
  const n=s._extract(p,image.width,image.height);
  if (!Number.isInteger(n) || n<0 || n>20000) throw Error('SIFT extraction failed or exceeded memory budget');
  const out=s._get_out(); // _extract may grow memory, so reacquire HEAPF32 afterward.
  if (!Number.isSafeInteger(out) || out<0 || out%4 || out+n*130*4>s.HEAPF32.byteLength) throw Error('Invalid SIFT result buffer');
  const a=s.HEAPF32.slice(out/4,out/4+n*130);
  for (let i=0;i<n;i++) {a[i*130]=(a[i*130]+.5)*image.sx-.5;a[i*130+1]=(a[i*130+1]+.5)*image.sy-.5;}
  return {n,a};
 } finally {s._free(p);}
}
export function siftSeeds(s: any, cv: any, ctx: any, log = (_: string) => {}) {
 const features=(g:ArrayLike<number>,w:number,h:number)=>extractSiftFeatures(s,g,w,h);
 const S=256, ref=Float32Array.from({length:S*S},(_,i)=>ctx.ref[Math.floor(i/S/16)][Math.floor(i%S/16)]);
 const a=features(ref,S,S), b=features(ctx.g,ctx.w,ctx.h);
 log(`SIFT WASM: ${a.n} texture / ${b.n} image features`);
 let P:number[][]=[], Q:number[][]=[];
 for(let i=0;i<b.n;i++) {
  const bi = i * 130 + 2;
  let best=Infinity,second=Infinity,jbest=0;
  for(let j=0;j<a.n;j++) {
   const aj = j * 130 + 2;
   let d = 0;
   for(let k=0;k<128;k++) {
    const v = b.a[bi+k] - a.a[aj+k];
    d += v * v;
    if(d >= second) break;
   }
   if(d < best) { second = best; best = d; jbest = j; }
   else if(d < second) second = d;
  }
  if(best < (ctx.opts.siftRatio ?? 0.68) ** 2 * second) { P.push([a.a[jbest*130],a.a[jbest*130+1]]); Q.push([b.a[i*130],b.a[i*130+1]]); }
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
