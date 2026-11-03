#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "solver_common.h"
#include <cstring>
#include <cstdlib>
struct Params {uint64_t base,ny,nz;uint32_t count,nb;int32_t xmin,ymin,zmin;uint32_t items,threads,fast,yoff,zoff;int32_t xbase;};
static const char*source=R"MSL(#include <metal_stdlib>
using namespace metal;
typedef uint uint32_t;typedef int int32_t;typedef ulong uint64_t;typedef long int64_t;typedef uchar uint8_t;
struct Block {int x,y,z,r,side;};
struct Params {ulong base,ny,nz;uint count,nb;int xmin,ymin,zmin;uint items,threads,fast,yoff,zoff;int xbase;};
ulong rotation_xz(int x,int z){
 int xp=as_type<int>(uint(x)*3129871u);
 return ulong(int64_t(xp))^(ulong(int64_t(z))*116129781ULL);
}
uint rotation_y(ulong xz,int y){
 ulong l=xz^ulong(int64_t(y));
 l=l*l*42317861ULL+l*11ULL;
 ulong seed=(((l>>16)^0x5DEECE66DULL)*0x5DEECE66DULL+11ULL)&0xFFFFFFFFFFFFULL;
 return uint(seed>>46);
}
uint2 rotation_y2(ulong xz,int y0,int y1){
 ulong l0=xz^ulong(int64_t(y0)),l1=xz^ulong(int64_t(y1));
 l0=l0*l0*42317861ULL+l0*11ULL;
 l1=l1*l1*42317861ULL+l1*11ULL;
 ulong s0=(((l0>>16)^0x5DEECE66DULL)*0x5DEECE66DULL+11ULL)&0xFFFFFFFFFFFFULL;
 ulong s1=(((l1>>16)^0x5DEECE66DULL)*0x5DEECE66DULL+11ULL)&0xFFFFFFFFFFFFULL;
 return uint2(uint(s0>>46),uint(s1>>46));
}
uint4 rotation_y4(ulong xz,int y0,int y1,int y2,int y3){
 ulong l0=xz^ulong(int64_t(y0)),l1=xz^ulong(int64_t(y1)),l2=xz^ulong(int64_t(y2)),l3=xz^ulong(int64_t(y3));
 l0=l0*l0*42317861ULL+l0*11ULL;
 l1=l1*l1*42317861ULL+l1*11ULL;
 l2=l2*l2*42317861ULL+l2*11ULL;
 l3=l3*l3*42317861ULL+l3*11ULL;
 ulong s0=(((l0>>16)^0x5DEECE66DULL)*0x5DEECE66DULL+11ULL)&0xFFFFFFFFFFFFULL;
 ulong s1=(((l1>>16)^0x5DEECE66DULL)*0x5DEECE66DULL+11ULL)&0xFFFFFFFFFFFFULL;
 ulong s2=(((l2>>16)^0x5DEECE66DULL)*0x5DEECE66DULL+11ULL)&0xFFFFFFFFFFFFULL;
 ulong s3=(((l3>>16)^0x5DEECE66DULL)*0x5DEECE66DULL+11ULL)&0xFFFFFFFFFFFFULL;
 return uint4(uint(s0>>46),uint(s1>>46),uint(s2>>46),uint(s3>>46));
}
uint rotation(int x,int y,int z){
 return rotation_y(rotation_xz(x,z),y);
}
bool matched(device const Block*blocks,uint nb,int x,int y,int z,uint start=0){
 for(uint i=start;i<nb;i++){
  Block b=blocks[i];uint r=rotation(x+b.x,y+b.y,z+b.z);
  if((r&(b.side?1u:3u))!=uint(b.r))return false;
 }
 return true;
}
void emit(device uint*results,uint index){uint slot=atomic_fetch_add_explicit((device atomic_uint*)results,1u,memory_order_relaxed);results[slot+1]=index;}
kernel void search(device const Block*blocks [[buffer(0)]],device uint*results [[buffer(1)]],constant Params&p [[buffer(2)]],uint tid [[thread_position_in_grid]]){
 for(uint item=0;item<p.items;item++){
  uint task=tid+item*p.threads;
  if(p.fast){
   uint ny=uint(p.ny),nz=uint(p.nz);
   uint columns=(p.count+p.yoff+ny-1u)/ny;if(task>=columns)break;
   uint zid=task+p.zoff;uint zz=zid%nz;
   int z=int(int64_t(p.zmin)+zz),x=int(int64_t(p.xbase)+zid/nz);
   uint begin=task==0?p.yoff:0u;
   uint end=min(ny,p.count+p.yoff-task*ny);
   Block b0=blocks[0];
   uint mask0=b0.side?1u:3u;
   uint target0=uint(b0.r);
   ulong xz0=rotation_xz(x+b0.x,z+b0.z);
   Block b1;uint mask1=0,target1=0;ulong xz1=0;
   if(p.nb>1){
    b1=blocks[1];
    mask1=b1.side?1u:3u;
    target1=uint(b1.r);
    xz1=rotation_xz(x+b1.x,z+b1.z);
   }
   Block b2;uint mask2=0,target2=0;ulong xz2=0;
   if(p.nb>2){
    b2=blocks[2];
    mask2=b2.side?1u:3u;
    target2=uint(b2.r);
    xz2=rotation_xz(x+b2.x,z+b2.z);
   }
   Block b3;uint mask3=0,target3=0;ulong xz3=0;
   if(p.nb>3){
    b3=blocks[3];
    mask3=b3.side?1u:3u;
    target3=uint(b3.r);
    xz3=rotation_xz(x+b3.x,z+b3.z);
   }
   uint yy=begin;
   for(;yy+1u<end;yy+=2u){
    int yA=int(int64_t(p.ymin)+yy);
    int yB=yA+1;
    uint2 r0=rotation_y2(xz0,yA+b0.y,yB+b0.y);
    bool passA=(r0.x&mask0)==target0;
    bool passB=(r0.y&mask0)==target0;
    if(!passA&&!passB)continue;
    if(p.nb>1){
     uint2 r1=rotation_y2(xz1,yA+b1.y,yB+b1.y);
     passA=passA&((r1.x&mask1)==target1);
     passB=passB&((r1.y&mask1)==target1);
     if(!passA&&!passB)continue;
    }
    if(p.nb>2){
     uint2 r2=rotation_y2(xz2,yA+b2.y,yB+b2.y);
     passA=passA&((r2.x&mask2)==target2);
     passB=passB&((r2.y&mask2)==target2);
     if(!passA&&!passB)continue;
    }
    if(p.nb>3){
     uint2 r3=rotation_y2(xz3,yA+b3.y,yB+b3.y);
     passA=passA&((r3.x&mask3)==target3);
     passB=passB&((r3.y&mask3)==target3);
     if(!passA&&!passB)continue;
    }
    if(passA){
     bool ok=true;
     if(p.nb>4){
      if(p.nb<=8){
       #pragma unroll
       for(uint i=4;i<8;i++){
        if(i>=p.nb)break;
        Block b=blocks[i];
        uint r=rotation(x+b.x,yA+b.y,z+b.z);
        if((r&(b.side?1u:3u))!=uint(b.r)){ok=false;break;}
       }
      }else{
       ok=matched(blocks,p.nb,x,yA,z,4);
      }
     }
     if(ok)emit(results,task*ny+yy-p.yoff);
    }
    if(passB){
     bool ok=true;
     if(p.nb>4){
      if(p.nb<=8){
       #pragma unroll
       for(uint i=4;i<8;i++){
        if(i>=p.nb)break;
        Block b=blocks[i];
        uint r=rotation(x+b.x,yB+b.y,z+b.z);
        if((r&(b.side?1u:3u))!=uint(b.r)){ok=false;break;}
       }
      }else{
       ok=matched(blocks,p.nb,x,yB,z,4);
      }
     }
     if(ok)emit(results,task*ny+(yy+1u)-p.yoff);
    }
   }
   for(;yy<end;yy++){
    int y=int(int64_t(p.ymin)+yy);
    uint r0=rotation_y(xz0,y+b0.y);
    if((r0&mask0)!=target0)continue;
    if(p.nb>1){
     uint r1=rotation_y(xz1,y+b1.y);
     if((r1&mask1)!=target1)continue;
    }
    if(p.nb>2){
     uint r2=rotation_y(xz2,y+b2.y);
     if((r2&mask2)!=target2)continue;
    }
    if(p.nb>3){
     uint r3=rotation_y(xz3,y+b3.y);
     if((r3&mask3)!=target3)continue;
    }
    bool ok=true;
    if(p.nb>4){
     if(p.nb<=8){
      #pragma unroll
      for(uint i=4;i<8;i++){
       if(i>=p.nb)break;
       Block b=blocks[i];
       uint r=rotation(x+b.x,y+b.y,z+b.z);
       if((r&(b.side?1u:3u))!=uint(b.r)){ok=false;break;}
      }
     }else{
      ok=matched(blocks,p.nb,x,y,z,4);
     }
    }
    if(ok)emit(results,task*ny+yy-p.yoff);
   }
  }else{
   uint index=task;if(index>=p.count)break;
   ulong id=p.base+index;
   int y=int(int64_t(p.ymin)+int64_t(id%p.ny));id/=p.ny;
   int z=int(int64_t(p.zmin)+int64_t(id%p.nz));id/=p.nz;
   int x=int(int64_t(p.xmin)+int64_t(id));
   if(matched(blocks,p.nb,x,y,z))emit(results,index);
  }
 }
})MSL";

struct Tune {uint32_t width=128,items=1;};
static uint64_t hashBytes(uint64_t h,const void*p,size_t n){
 const auto*b=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;i++){h^=b[i];h*=1099511628211ULL;}return h;
}
static id<MTLCommandBuffer> launch(id<MTLCommandQueue>queue,id<MTLComputePipelineState>pipe,
 id<MTLBuffer>bb,id<MTLBuffer>bf,Params p,Tune t){
 p.items=t.items;p.threads=(p.count+t.items-1)/t.items;
 p.fast=p.ny<=4096&&p.nz<=UINT32_MAX-4194304u&&!std::getenv("GRASS_METAL_FORCE_WIDE");
 uint64_t origin=p.base;p.yoff=uint32_t(origin%p.ny);origin/=p.ny;p.zoff=uint32_t(origin%p.nz);origin/=p.nz;
 p.xbase=int32_t(int64_t(p.xmin)+int64_t(origin));
 uint32_t tasks=p.fast?uint32_t((p.count+p.yoff+p.ny-1)/p.ny):p.count;
 p.threads=(tasks+t.items-1)/t.items;
 *static_cast<uint32_t*>(bf.contents)=0;
 id<MTLCommandBuffer>cmd=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cmd computeCommandEncoder];
 if(!cmd||!enc)throw std::runtime_error("Metal encoder allocation failed");
 [enc setComputePipelineState:pipe];[enc setBuffer:bb offset:0 atIndex:0];[enc setBuffer:bf offset:0 atIndex:1];[enc setBytes:&p length:sizeof(p) atIndex:2];
 [enc dispatchThreads:MTLSizeMake(p.threads,1,1) threadsPerThreadgroup:MTLSizeMake(t.width,1,1)];
 [enc endEncoding];[cmd commit];
 return cmd;
}
// Timings include only the GPU interval where available. Correctness comparisons and
// command construction are outside that interval. Wall time is the older-OS fallback.
static double dispatch(id<MTLCommandQueue>queue,id<MTLComputePipelineState>pipe,
 id<MTLBuffer>bb,id<MTLBuffer>bf,Params p,Tune t){
 @autoreleasepool {
 auto start=std::chrono::steady_clock::now();
 id<MTLCommandBuffer>cmd=launch(queue,pipe,bb,bf,p,t);
 [cmd waitUntilCompleted];
 if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cmd.error?[[cmd.error localizedDescription] UTF8String]:"Metal dispatch failed");
 double gpu=cmd.GPUEndTime-cmd.GPUStartTime;
 return gpu>0?gpu:std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 }
}
static std::vector<uint32_t> resultIds(id<MTLBuffer>bf,uint32_t count){
 auto*r=static_cast<uint32_t*>(bf.contents);if(r[0]>count)throw std::runtime_error("GPU result capacity exceeded");
 std::vector<uint32_t>ids(r+1,r+1+r[0]);std::sort(ids.begin(),ids.end());
 for(size_t i=0;i<ids.size();i++)if(ids[i]>=count||(i&&ids[i]==ids[i-1]))throw std::runtime_error("Invalid/duplicate GPU result index");
 return ids;
}
static NSString* cachePath(){
 const char*env=std::getenv("GRASS_METAL_AUTOTUNE_CACHE");if(env&&*env)return [NSString stringWithUTF8String:env];
 return [[NSSearchPathForDirectoriesInDomains(NSCachesDirectory,NSUserDomainMask,YES) firstObject] stringByAppendingPathComponent:@"grass-solver/metal-autotune-v2.plist"];
}
static Tune autotune(id<MTLDevice>device,id<MTLCommandQueue>queue,id<MTLComputePipelineState>pipe,
 id<MTLBuffer>bb,id<MTLBuffer>bf,const std::vector<Block>&blocks,const Options&o,const std::string&mode,NSString*path){
 Tune fallback{uint32_t(std::min<NSUInteger>(128,pipe.maxTotalThreadsPerThreadgroup)),1};
 if(mode=="off"){std::cerr<<"autotune off width "<<fallback.width<<" items 1\n";return fallback;}
 // Exact observations and search shape matter: early rejection and integer division
 // costs change with the formation, dimensions and origin. Kernel source invalidates
 // stale results, as does GPU identity/architecture and the supported SIMD width.
 uint64_t h=hashBytes(14695981039346656037ULL,source,std::strlen(source));
 h=hashBytes(h,blocks.data(),blocks.size()*sizeof(Block));
 h=hashBytes(h,o.dim,sizeof(o.dim));h=hashBytes(h,o.lo,sizeof(o.lo));bool wide=std::getenv("GRASS_METAL_FORCE_WIDE")!=nullptr;h=hashBytes(h,&wide,sizeof(wide));
 NSString*key=[NSString stringWithFormat:@"columns-v3|%@|%@|%@|%lu|%lu|%016llx",device.name,device.architecture.name,
 NSProcessInfo.processInfo.operatingSystemVersionString,(unsigned long)pipe.threadExecutionWidth,
 (unsigned long)pipe.maxTotalThreadsPerThreadgroup,(unsigned long long)h];
 NSMutableDictionary*cache=[[NSDictionary dictionaryWithContentsOfFile:path] mutableCopy];if(!cache)cache=[NSMutableDictionary dictionary];
 NSDictionary*record=cache[key];
 if(mode=="auto"&&[record isKindOfClass:[NSDictionary class]]){
  id w=record[@"width"],i=record[@"items"],ms=record[@"milliseconds"];
  if([w isKindOfClass:[NSNumber class]]&&[i isKindOfClass:[NSNumber class]]&&[ms isKindOfClass:[NSNumber class]]){
   uint32_t width=[w unsignedIntValue],items=[i unsignedIntValue];
   if((width==64||width==128||width==256||width==512)&&width<=pipe.maxTotalThreadsPerThreadgroup&&
      width%pipe.threadExecutionWidth==0&&(items==1||items==2||items==4||items==8)&&[ms doubleValue]>0){
    std::cerr<<"autotune cache hit width "<<width<<" items "<<items<<" measured_ms "<<[ms doubleValue]<<" cache "<<path.UTF8String<<"\n";return {width,items};
   }
  }
 }
 if(mode=="auto"&&o.total<32768){std::cerr<<"autotune skipped tiny search width "<<fallback.width<<" items 1\n";return fallback;}
 auto start=std::chrono::steady_clock::now();
 uint32_t count=uint32_t(std::min<uint64_t>(262144,o.total));
 std::vector<uint64_t>bases{0};if(o.total>count){bases.push_back((o.total-count)/2);bases.push_back(o.total-count);}
 std::vector<std::vector<uint32_t>>expected;
 auto params=[&](uint64_t base){return Params{base,o.dim[1],o.dim[2],count,uint32_t(blocks.size()),o.lo[0],o.lo[1],o.lo[2],1,count,0,0,0,0};};
 for(auto base:bases){dispatch(queue,pipe,bb,bf,params(base),fallback);expected.push_back(resultIds(bf,count));}
 Tune best=fallback;double bestTime=std::numeric_limits<double>::max();int tested=0;
 std::vector<Tune>variants;
 for(uint32_t width:{64u,128u,256u,512u})for(uint32_t items:{1u,2u,4u,8u})
  if(width<=pipe.maxTotalThreadsPerThreadgroup&&width%pipe.threadExecutionWidth==0)variants.push_back({width,items});
 std::vector<std::vector<double>>times(variants.size());
 // Warm every variant before measuring. Interleave a permuted candidate order in
 // each round to reduce first-candidate/DVFS bias, then choose the median round.
 for(auto candidate:variants)dispatch(queue,pipe,bb,bf,params(bases[0]),candidate);
 for(size_t round=0;round<3;round++)for(size_t k=0;k<variants.size();k++){
  size_t v=(k*5+round*7)%variants.size();double seconds=0;
  for(size_t j=0;j<bases.size();j++){
   std::memset(bf.contents,0xff,(size_t(count)+1)*4); // Poison indices to detect missing writes.
   seconds+=dispatch(queue,pipe,bb,bf,params(bases[j]),variants[v]);
   if(resultIds(bf,count)!=expected[j])throw std::runtime_error("Autotune variant correctness mismatch");
  }
  times[v].push_back(seconds/bases.size());
 }
 for(size_t v=0;v<variants.size();v++){
  std::sort(times[v].begin(),times[v].end());double seconds=times[v][1];tested++;
  std::cerr<<"autotune sample width "<<variants[v].width<<" items "<<variants[v].items<<" gpu_ms "<<seconds*1000<<"\n";
  if(seconds<bestTime){bestTime=seconds;best=variants[v];}
 }
 if(!tested)throw std::runtime_error("No valid Metal autotune variants");
 cache[key]=@{@"width":@(best.width),@"items":@(best.items),@"milliseconds":@(bestTime*1000)};
 // Bound metadata growth. A cache failure never prevents a valid search.
 if(cache.count>64){[cache removeAllObjects];cache[key]=@{@"width":@(best.width),@"items":@(best.items),@"milliseconds":@(bestTime*1000)};}
 NSError*err=nil;BOOL dir=[[NSFileManager defaultManager] createDirectoryAtPath:path.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:&err];
 BOOL saved=dir&&[cache writeToFile:path atomically:YES];
 std::cerr<<"autotune selected width "<<best.width<<" items "<<best.items<<" measured_ms "<<bestTime*1000
 <<" variants "<<tested<<" verified_samples "<<tested*3*bases.size()<<" tune_seconds "
 <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" cache "<<path.UTF8String<<(saved?" saved":" not-saved")<<"\n";
 return best;
}
int main(int argc,char**argv){@autoreleasepool{try{
 std::string mode="auto";NSString*path=cachePath();bool clear=false;std::vector<char*>args{argv[0]};
 for(int i=1;i<argc;i++){
  std::string a=argv[i];
  if(a=="--autotune"){if(++i==argc)throw std::runtime_error("Missing autotune mode");mode=argv[i];}
  else if(a=="--autotune-cache"){if(++i==argc)throw std::runtime_error("Missing cache path");path=[NSString stringWithUTF8String:argv[i]];}
  else if(a=="--clear-autotune-cache")clear=true;
  else {if(a=="--help"||a=="-h")std::cout<<"Metal options: --autotune auto|force|off (default auto), --autotune-cache FILE, --clear-autotune-cache\n";args.push_back(argv[i]);}
 }
 if(mode!="auto"&&mode!="force"&&mode!="off")throw std::runtime_error("autotune must be auto, force or off");
 auto o=options(int(args.size()),args.data());auto original=load(o.path);
 for(int n:o.north)orient(original,n,o);
 if(clear){NSError*error=nil;if([[NSFileManager defaultManager] fileExistsAtPath:path]&&![[NSFileManager defaultManager] removeItemAtPath:path error:&error])throw std::runtime_error("Cannot clear autotune cache");std::cerr<<"autotune cache cleared "<<path.UTF8String<<"\n";}
 id<MTLDevice>device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("No Metal device");
 NSError*error=nil;
 id<MTLLibrary>lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&error];
 if(!lib)throw std::runtime_error([[error localizedDescription] UTF8String]);
 id<MTLComputePipelineState>pipe=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"search"] error:&error];
 if(!pipe)throw std::runtime_error([[error localizedDescription] UTF8String]);
 id<MTLCommandQueue>queue=[device newCommandQueue];
 const uint32_t batch=4194304;
 id<MTLBuffer>bf[2]={
  [device newBufferWithLength:(size_t(batch)+1)*4 options:MTLResourceStorageModeShared],
  [device newBufferWithLength:(size_t(batch)+1)*4 options:MTLResourceStorageModeShared]
 };
 if(!queue||!bf[0]||!bf[1])throw std::runtime_error("Metal allocation failed");
 double searchSeconds=0;uint64_t done=0,matches=0;auto start=std::chrono::steady_clock::now();auto last=start;
 std::signal(SIGINT,stop_search);std::signal(SIGTERM,stop_search);
 for(int north:o.north){
  auto blocks=orient(original,north,o);
  std::stable_sort(blocks.begin(),blocks.end(),[](const Block&a,const Block&b){return a.side<b.side;});
  id<MTLBuffer>bb=[device newBufferWithBytes:blocks.data() length:blocks.size()*sizeof(Block) options:MTLResourceStorageModeShared];if(!bb)throw std::runtime_error("Metal block allocation failed");
  Tune tune=autotune(device,queue,pipe,bb,bf[0],blocks,o,mode,path);
  auto searchStart=std::chrono::steady_clock::now();
  uint64_t base=0;uint32_t bufIdx=0;
  id<MTLCommandBuffer>inflightCmd=nil;
  uint32_t inflightBuf=0;uint64_t inflightBase=0;uint32_t inflightCount=0;
  if(base<o.total&&!stopped){
   inflightCount=uint32_t(std::min<uint64_t>(batch,o.total-base));
   Params p{base,o.dim[1],o.dim[2],inflightCount,uint32_t(blocks.size()),o.lo[0],o.lo[1],o.lo[2],tune.items,0,0,0,0,0};
   inflightCmd=launch(queue,pipe,bb,bf[bufIdx],p,tune);
   inflightBuf=bufIdx;inflightBase=base;
   base+=inflightCount;bufIdx^=1;
  }
  while(inflightCmd!=nil){
   id<MTLCommandBuffer>nextCmd=nil;
   uint32_t nextBuf=0;uint64_t nextBase=0;uint32_t nextCount=0;
   if(base<o.total&&!stopped){
    nextCount=uint32_t(std::min<uint64_t>(batch,o.total-base));
    Params p{base,o.dim[1],o.dim[2],nextCount,uint32_t(blocks.size()),o.lo[0],o.lo[1],o.lo[2],tune.items,0,0,0,0,0};
    nextCmd=launch(queue,pipe,bb,bf[bufIdx],p,tune);
    nextBuf=bufIdx;nextBase=base;
    base+=nextCount;bufIdx^=1;
   }
   [inflightCmd waitUntilCompleted];
   if(inflightCmd.status!=MTLCommandBufferStatusCompleted)
    throw std::runtime_error(inflightCmd.error?[[inflightCmd.error localizedDescription] UTF8String]:"Metal dispatch failed");
   auto ids=resultIds(bf[inflightBuf],inflightCount);
   for(uint32_t i:ids){int32_t x,y,z;decode(inflightBase+i,o,x,y,z);std::cout<<x<<" "<<y<<" "<<z<<" "<<north<<"\n";matches++;}
   if(!std::cout)throw std::runtime_error("Output write failed");
   done+=inflightCount;
   auto now=std::chrono::steady_clock::now();
   if(std::chrono::duration<double>(now-last).count()>=1){double sec=std::chrono::duration<double>(now-start).count();std::cerr<<"progress "<<done<<"/"<<o.total*o.north.size()<<" rate "<<double(done)/sec<<" candidates/s matches "<<matches<<"\n";last=now;}
   if(stopped){if(nextCmd)[nextCmd waitUntilCompleted];break;}
   inflightCmd=nextCmd;inflightBuf=nextBuf;inflightBase=nextBase;inflightCount=nextCount;
  }
  searchSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-searchStart).count();
  if(stopped)break;
 }
 double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::cout.flush();
 std::cerr<<"search_loop seconds "<<searchSeconds<<" rate "<<double(done)/std::max(searchSeconds,1e-9)<<" candidates/s tuning_excluded 1\n";
 std::cerr<<(stopped?"cancelled":"complete")<<" total "<<done<<" matches "<<matches<<" seconds "<<sec<<" rate "<<double(done)/std::max(sec,1e-9)<<" candidates/s\n";
 return stopped?130:0;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}}
