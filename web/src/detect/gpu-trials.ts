import type {Block} from './webgpu-pipeline';
// Same fixed configuration and anchored union as native. No WASM calls.
export const GPU_TRIALS = [
 ['original',{}],['relaxed',{accept:.84,relaxed:.72,margin:.30}],
 ['refine',{refineMin:.62,refineMargin:.20}],
 ['refit-refine',{refineMin:.62,refineMargin:.20,refitPlane:true}],
 ['ratio',{siftRatio:.76}],
 ['combo',{accept:.84,relaxed:.72,margin:.30,refineMin:.62,refineMargin:.20,refitPlane:true}],
 ['flat-ratio',{refitFlat:true,siftRatio:.76}],
 ['rescue',{refitFlat:true,refineMin:.55,refineMargin:.10,refineVotes:2,refineShift:6}],
 ['ratio80',{siftRatio:.80}],
] as const;
export function mergeGpuAligned(a:Block[], incoming:Block[]):Block[]{
 const votes=new Map<string,{offset:number[],count:number}>();let evidence=0;
 for(const b of incoming){let nearest:Block|undefined;let best=Infinity;
  for(const o of a){const scale=Math.min(...o.quad.map((p,i)=>Math.hypot(p[0]-o.quad[(i+1)%4][0],p[1]-o.quad[(i+1)%4][1])));const error=b.quad.reduce((s,p,i)=>s+Math.hypot(p[0]-o.quad[i][0],p[1]-o.quad[i][1]),0)/4;if(error<Math.max(3,scale*.12)&&error<best){nearest=o;best=error;}}
  if(!nearest)continue;if(nearest.rotation!==b.rotation)return a;
  const offset=[nearest.x-b.x,nearest.y-b.y,nearest.z-b.z];const key=offset.join(',');const v=votes.get(key);if(v)v.count++;else votes.set(key,{offset,count:1});evidence++;
 }
 let winner:{offset:number[],count:number}|undefined;for(const v of votes.values())if(!winner||v.count>winner.count)winner=v;
 if(!winner||winner.count<3||winner.count<evidence*.9)return a;
 const shifted=incoming.map(b=>({...b,x:b.x+winner!.offset[0],y:b.y+winner!.offset[1],z:b.z+winner!.offset[2]}));
 const key=(b:Block)=>[b.x,b.y,b.z].join(',');const coords=new Map(a.map(b=>[key(b),b.rotation]));
 for(const b of shifted)if(coords.has(key(b))&&coords.get(key(b))!==b.rotation)return a;
 const merged=a.slice();for(const b of shifted)if(!coords.has(key(b))){merged.push(b);coords.set(key(b),b.rotation);}return merged;
}
/** Native top-down spectral size proposals, JS transport geometry, GPU scoring only. */
export function topDownSizes(data:Uint8Array,w:number,h:number):number[]{
 const periods:number[]=[];
 for(let axis=0;axis<2;axis++){const n=axis?h:w,other=axis?w:h;const e=new Float64Array(n-1);
  for(let j=0;j<other;j+=3)for(let i=0;i<n-1;i++){const p=axis?i*w+j:j*w+i;e[i]+=Math.abs(data[4*(p+(axis?w:1))+1]-data[4*p+1]);}
  const mean=e.reduce((a,b)=>a+b,0)/e.length;const peaks:{period:number,power:number}[]=[];
  for(let k=Math.ceil(n/24);k<n/2;k++){let re=0,im=0;const angle=2*Math.PI*k/e.length;for(let i=0;i<e.length;i++){const v=e[i]-mean;re+=v*Math.cos(angle*i);im+=v*Math.sin(angle*i);}peaks.push({period:e.length/k*16,power:re*re+im*im});}
  peaks.sort((a,b)=>b.power-a.power);periods.push(...peaks.slice(0,2).map(p=>p.period));
 }
 const sizes:number[]=[];for(const p of periods)for(const d of [-1,0,1]){const n=Math.floor(p+.5)+d;if(n>=48&&n<Math.min(w,h)*.6&&!sizes.includes(n))sizes.push(n);}return sizes;
}
