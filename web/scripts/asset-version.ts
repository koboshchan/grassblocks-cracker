import {createHash} from 'node:crypto';
/** Content-hash directory for a set of files that must be loaded as one unit. Deterministic. */
export function versionedAssets(files:Record<string,Uint8Array>){
 const h=createHash('sha256');
 for(const n of Object.keys(files).sort()){h.update(n+'\0');h.update(createHash('sha256').update(files[n]).digest());}
 const hash=h.digest('hex').slice(0,16);
 return {hash,dir:'assets/'+hash+'/'};
}
