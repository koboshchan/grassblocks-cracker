/** Transport/validation only. All image and geometry algorithms run in WASM. */
export type PipelineModule = {
 HEAPU8: Uint8Array; HEAPF64: Float64Array;
 _malloc(n:number):number; _free(p:number):void;
 _grass_detect(p:number,w:number,h:number,r:number,o:number,s:number,n:number,a:number):number;
 _grass_classify(p:number,w:number,h:number,r:number,q:number):number;
 _grass_result():number; UTF8ToString(p:number):string;
};
let native:PipelineModule|null=null;
export function setPipeline(module:PipelineModule){native=module;}
export const detectorBackend='WASM native pipeline (OpenCV + VLFeat + SIMD)';
function module(){if(!native)throw Error('Native WASM detection pipeline unavailable');return native;}
function alloc(bytes:Uint8Array|Float64Array,owned:number[]){const m=module(),p=m._malloc(bytes.byteLength||8);if(!p)throw Error('Native pipeline allocation failed');owned.push(p);if(bytes instanceof Float64Array)m.HEAPF64.set(bytes,p/8);else m.HEAPU8.set(bytes,p);return p;}
function reference(ref:unknown){if(!Array.isArray(ref)||ref.length!==16||ref.some(row=>!Array.isArray(row)||row.length!==16))throw Error('Expected 16x16 reference');const values=new Float64Array(ref.flat());if(values.some(v=>!Number.isFinite(v)||v<0||v>255))throw Error('Invalid reference pixels');return values;}
function quad(q:unknown){if(!Array.isArray(q)||q.length!==4||q.some(p=>!Array.isArray(p)||p.length!==2))throw Error('Expected four 2D quad corners');const a=new Float64Array(q.flat());if(a.some(v=>!Number.isFinite(v)))throw Error('Nonfinite quad corner');return a;}
const defaults={accept:.88,acceptMargin:.35,relaxed:.78,coarse:.55,refineMin:-1,refineMargin:-1,refineShift:3.5,siftRatio:.68,refineVotes:3,refitPlane:false,refitFlat:false};
function options(opts:Record<string,unknown>={}){const supported=new Set(Object.keys(defaults));if(Object.keys(opts).some(k=>!supported.has(k)))throw Error('Unsupported native detection option');const o={...defaults,...opts};const a=new Float64Array([Number(o.accept),Number(o.acceptMargin),Number(o.relaxed),Number(o.coarse),Number(o.refineMin),Number(o.refineMargin),Number(o.refineShift),Number(o.siftRatio),Number(o.refineVotes),Number(!!o.refitPlane),Number(!!o.refitFlat),0]);if(a.some(v=>!Number.isFinite(v)))throw Error('Nonfinite detection option');return a;}
function result(){const m=module(),r=JSON.parse(m.UTF8ToString(m._grass_result()));if(r.error)throw Error(r.error);return r;}
export function classifyQuad(rgba:Uint8Array,w:number,h:number,ref:unknown,q:unknown){const m=module(),owned:number[]=[];try{const p=alloc(rgba,owned),r=alloc(reference(ref),owned),s=alloc(quad(q),owned);m._grass_classify(p,w,h,r,s);return result();}finally{owned.forEach(p=>m._free(p));}}
export function detect(rgba:Uint8Array,w:number,h:number,ref:unknown,opts:Record<string,unknown>={},seeds:unknown[]=[]){const m=module(),owned:number[]=[];if(!Array.isArray(seeds)||seeds.length>4096)throw Error('Invalid seed list');const packed=new Float64Array(seeds.flatMap(q=>Array.from(quad(q))));const custom=Object.entries(opts).some(([key,value])=>key!=='accept'||value!==.88);try{const p=alloc(rgba,owned),r=alloc(reference(ref),owned),o=alloc(options(opts),owned),s=alloc(packed,owned);m._grass_detect(p,w,h,r,o,s,seeds.length,Number(!seeds.length&&!custom));return result();}finally{owned.forEach(p=>m._free(p));}}
