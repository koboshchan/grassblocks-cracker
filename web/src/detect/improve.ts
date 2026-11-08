// @ts-nocheck
import {runDetection} from './detect';

// Local research entry point, not wired into the deployed application.
// Fixed pixel-only trials. Never select a trial using a hash/coordinate answer.
export const IMPROVEMENT_TRIALS = [
  ['original', {}],
  ['relaxed', {accept:.84, relaxed:.72, acceptMargin:.30}],
  ['refine', {refineMin:.62, refineMargin:.20}],
  ['refit-refine', {refitPlane:true,refineMin:.62,refineMargin:.20}],
  ['ratio', {siftRatio:.76}],
];

export function runImprovedDetection(cv,img,ref,progress=()=>{},hooks={}) {
  const seedCache=new Map(), trials=[];
  const cached={...hooks};
  if(hooks.featureSeeds) cached.featureSeeds=(cv,ctx,log)=>{
    const ratio=ctx.opts.siftRatio ?? .68;
    if(!seedCache.has(ratio)) seedCache.set(ratio,hooks.featureSeeds(cv,ctx,log));
    return seedCache.get(ratio);
  };
  let best=null;
  for(const [name,opts] of IMPROVEMENT_TRIALS) {
    progress('trial '+name);
    const result=runDetection(cv,img,ref,opts,progress,cached);
    trials.push({name,tiles:result.blocks.length,notes:result.notes});
    // Strictly greater keeps the original on ties and never reduces tile count.
    // This is a support heuristic, not a proof that extra observations are correct.
    if(!best || result.blocks.length>best.blocks.length) best={...result,trial:name};
  }
  return {...best,trials};
}
