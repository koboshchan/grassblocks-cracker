export interface Block { x:number; y:number; z:number; rotation:number; is_side:number; [key:string]:unknown }
export interface SearchOptions { xmin:number; xmax:number; ymin:number; ymax:number; zmin:number; zmax:number; north:number[]; signal?:AbortSignal; maxMatches?:number; onProgress?:(p:{done:number;total:number;seconds:number;rate:number})=>void; onMatch?:(m:{x:number;y:number;z:number;north:number})=>void }
// Vanilla modern BlockPos#getSeed and Java Random.nextInt(4), integer exact.
export function rotationBigInt(x:number,y:number,z:number,side=0) {
 let l=BigInt(Math.imul(x,3129871)) ^ (BigInt(z)*116129781n) ^ BigInt(y);
 l=BigInt.asIntN(64,l*l*42317861n+l*11n);
 let s=((l>>16n)^0x5deece66dn)&((1n<<48n)-1n);
 s=(s*0x5deece66dn+11n)&((1n<<48n)-1n);
 const r=Number(s>>46n);return side?r%2:r;
}
export function mulhi(a:number,b:number) {
 const al=a&65535, ah=a>>>16, bl=b&65535,bh=b>>>16;
 const p0=al*bl,mid=al*bh+(p0>>>16),mid2=(mid&65535)+ah*bl;
 return (ah*bh+(mid>>>16)+(mid2>>>16))>>>0;
}
export function rotationPair(x:number,y:number,z:number,side=0) {
 const xp=Math.imul(x,3129871),zl=Math.imul(z,116129781);
 const zh=(mulhi(z>>>0,116129781)+(z<0?-116129781:0))>>>0;
 const lo=(xp^zl^y)>>>0, hi=((xp<0?0xffffffff:0)^zh^(y<0?0xffffffff:0))>>>0;
 const sql=Math.imul(lo,lo)>>>0,sqh=(mulhi(lo,lo)+Math.imul(2*lo,hi))>>>0;
 const pl=Math.imul(sql,42317861)>>>0,ph=(mulhi(sql,42317861)+Math.imul(sqh,42317861))>>>0;
 const ll=Math.imul(lo,11)>>>0,lh=(mulhi(lo,11)+Math.imul(hi,11))>>>0;
 const rl=(pl+ll)>>>0,rh=(ph+lh+(rl<pl?1:0))>>>0;
 const sl=(((rl>>>16)|(rh<<16))^0xdeece66d)>>>0,sh=((rh>>>16)^5)>>>0;
 const prod=Math.imul(sl,0xdeece66d)>>>0;
 const high=(mulhi(sl,0xdeece66d)+Math.imul(sh,0xdeece66d)+Math.imul(sl,5)+((prod+11)>0xffffffff?1:0))>>>0;
 const r=(high>>>14)&3;return side?r%2:r;
}
export function turnBlocks(blocks:Block[],n:number) {
 return blocks.map(b=>{let x=b.x,z=b.z;for(let j=0;j<n;j++)[x,z]=[-z,x];return {...b,x,z,rotation:(b.rotation+n)%(b.is_side?2:4)};});
}
export function validate(blocks:Block[],o:SearchOptions) {
 if(!blocks.length||blocks.length>512)throw Error('Use 1 to 512 block observations.');
 for(const b of blocks)for(const k of ['x','y','z','rotation','is_side'])if(!Number.isInteger(b[k]))throw Error('Formation values must be integers.');
 for(const b of blocks)if(Math.abs(b.x)>1000000||Math.abs(b.y)>1000000||Math.abs(b.z)>1000000||![0,1].includes(b.is_side)||b.rotation<0||b.rotation>=(b.is_side?2:4))throw Error('Invalid block observation.');
 for(const a of ['x','y','z']){const l=o[a+'min'],h=o[a+'max'];if(!Number.isInteger(l)||!Number.isInteger(h)||l>h||Math.abs(l)>30000000||Math.abs(h)>30000000)throw Error('Invalid inclusive search bounds.');}
 if(!Array.isArray(o.north)||!o.north.length||o.north.some(n=>!Number.isInteger(n)||n<0||n>3))throw Error('Choose north 0–3.');
 const total=(o.xmax-o.xmin+1)*(o.ymax-o.ymin+1)*(o.zmax-o.zmin+1)*o.north.length;
 if(!Number.isSafeInteger(total))throw Error('Search volume exceeds exact browser progress counter. Narrow the bounds.');
 return total;
}
