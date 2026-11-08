// @ts-nocheck
import {runDetection} from './detect';

// Pixel-only ensemble used by browser Auto detection and research benchmarks.
// Fixed pixel-only trials. Never select a trial using a hash/coordinate answer.
export const IMPROVEMENT_TRIALS = [
  ['original', {}],
  ['relaxed', {accept:.84, relaxed:.72, acceptMargin:.30}],
  ['refine', {refineMin:.62, refineMargin:.20}],
  ['refit-refine', {refitPlane:true,refineMin:.62,refineMargin:.20}],
  ['ratio', {siftRatio:.76}],
  ['combo', {accept:.84,relaxed:.72,acceptMargin:.30,refineMin:.62,refineMargin:.20,refitPlane:true}],
  ['flat-ratio', {refitFlat:true,siftRatio:.76}],
  ['rescue', {refitFlat:true,refineMin:.55,refineMargin:.10,refineVotes:2,refineShift:6}],
  ['ratio80', {siftRatio:.80}],
];

// Require repeated geometric agreement in the SAME corner convention.
// Do not guess rotations/heights between disconnected planes.
export function mergeAligned(original, incoming) {
  const evidence=[], votes=new Map();
  for(const b of incoming) {
    let nearest=null,best=Infinity;
    for(const a of original) {
      const scale=Math.min(...a.quad.map((p,i)=>Math.hypot(p[0]-a.quad[(i+1)%4][0],p[1]-a.quad[(i+1)%4][1])));
      const err=b.quad.reduce((s,p,i)=>s+Math.hypot(p[0]-a.quad[i][0],p[1]-a.quad[i][1]),0)/4;
      if(err<Math.max(3,scale*.12)&&err<best){best=err;nearest=a;}
    }
    if(!nearest)continue;
    const a=nearest,key=[a.x-b.x,a.y-b.y,a.z-b.z].join(',');
    if(a.rotation!==b.rotation)return original;
    evidence.push(key);votes.set(key,(votes.get(key)||0)+1);
  }
  const winner=[...votes].sort((a,b)=>b[1]-a[1])[0];
  if(!winner||winner[1]<3||winner[1]<evidence.length*.9)return original;
  const offset=winner[0].split(',').map(Number),merged=original.slice();
  const coords=new Map(original.map(b=>[[b.x,b.y,b.z].join(','),b]));
  // A translated coordinate with a conflicting rotation invalidates this trial.
  for(const b of incoming) {
    const key=[b.x+offset[0],b.y+offset[1],b.z+offset[2]].join(',');
    if(coords.has(key)&&coords.get(key).rotation!==b.rotation)return original;
  }
  for(const b of incoming) {
    const [x,y,z]=[b.x+offset[0],b.y+offset[1],b.z+offset[2]],key=[x,y,z].join(',');
    if(!coords.has(key)){const aligned={...b,x,y,z};merged.push(aligned);coords.set(key,aligned);}
  }
  return merged;
}

export function runImprovedDetection(cv,img,ref,progress: (text: string) => void=()=>{},hooks={}) {
  const seedCache=new Map(),trials=[];
  const cached={...hooks};
  if(hooks.featureSeeds) cached.featureSeeds=(cv,ctx,log)=>{
    const ratio=ctx.opts.siftRatio ?? .68;
    if(!seedCache.has(ratio)) seedCache.set(ratio,hooks.featureSeeds(cv,ctx,log));
    return seedCache.get(ratio);
  };
  let result=null,blocks=[];
  for(const [name,opts] of IMPROVEMENT_TRIALS) {
    progress('trial '+name);
    const started=performance.now();
    const candidate=runDetection(cv,img,ref,opts,progress,cached);
    if(!result){result=candidate;blocks=candidate.blocks;}
    else blocks=mergeAligned(blocks,candidate.blocks);
    trials.push({name,tiles:candidate.blocks.length,mergedTiles:blocks.length,seconds:(performance.now()-started)/1000,notes:candidate.notes});
    hooks.trialComplete?.({blocks,trials:trials.slice()});
  }
  return {...result,blocks,trial:'original-anchored geometric union',trials,
    notes:[...result.notes,'Additional trials registered by repeated pixel geometry; not coordinate search.'],
    // Merged quads may have separately refined calibrations.
    camera:null,planeH:null};
}
