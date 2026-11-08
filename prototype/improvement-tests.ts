import assert from 'node:assert/strict';
import {DEFAULTS, makeContext, classify} from '../web/src/detect/detect';
import {IMPROVEMENT_TRIALS,runImprovedDetection,mergeAligned} from '../web/src/detect/improve';
assert.deepEqual(DEFAULTS,{accept:.88,acceptMargin:.35,relaxed:.78,coarse:.55});
const ref=Array.from({length:16},(_,y)=>Array.from({length:16},(_,x)=>((x*73+y*97+x*y*19)%151)+20));
// Pixel-centred expanded pixels make the sampled patch exactly one texel per cell.
const data=new Uint8Array(17*17*4);
for(let y=0;y<17;y++)for(let x=0;x<17;x++)data[(y*17+x)*4+1]=ref[Math.min(y,15)][Math.min(x,15)];
const image={width:17,height:17,data};
const ctx=makeContext(image,ref),q=[[-.5,-.5],[15.5,-.5],[15.5,15.5],[-.5,15.5]];
// Shift corners inward infinitesimally to satisfy strict image bounds while sampling centres.
const quad=[[0,0],[16,0],[16,16],[0,16]];
const score=classify(ctx,quad);
const tinted=makeContext(image,ref);
tinted.g=Float32Array.from(ctx.g,v=>.5*v+35);
const after=classify(tinted,quad);
assert.equal(after.rotation,score.rotation);
assert.ok(Math.abs(after.score-score.score)<1e-6,'NCC already cancels affine biome tint/lighting');
assert.ok(!ctx.opts.refitPlane&&!ctx.opts.refitFlat&&!ctx.opts.siftRatio,'all new options are opt-in');
assert.equal(IMPROVEMENT_TRIALS[0][0],'original');
const missing=runImprovedDetection(null,image,ref);
assert.equal(missing.trial,'original-anchored geometric union');
assert.equal(missing.blocks.length,0);
assert.equal(missing.trials.length,IMPROVEMENT_TRIALS.length);
console.log('PASS unchanged defaults, affine tint invariance, opt-in options, original-on-tie, no-cv fallback');

const blocks=Array.from({length:3},(_,i)=>({x:i,y:0,z:0,rotation:i,quad:[[i*25,0],[i*25+20,0],[i*25+20,20],[i*25,20]]}));
const incoming=[...blocks.map(b=>({...b,x:b.x-10})),{x:-7,y:0,z:0,rotation:0,quad:[[75,0],[95,0],[95,20],[75,20]]}];
assert.equal(mergeAligned(blocks,incoming).length,4);
assert.equal(mergeAligned(blocks,incoming)[3].x,3);
assert.equal(mergeAligned(blocks,[...incoming.slice(0,2),incoming[3]]).length,3);
assert.equal(mergeAligned(blocks,incoming.map((b,i)=>i===0?{...b,rotation:3}:b)).length,3);
console.log('PASS repeated geometric registration, anchor preservation, weak/rotation-conflict rejection');
