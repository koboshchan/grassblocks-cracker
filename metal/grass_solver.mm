#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "solver_common.h"
#include <cstring>
#include <cstdlib>
struct Params {uint64_t base,ny,nz;uint32_t count,nb;int32_t xmin,ymin,zmin;uint32_t items,threads;};
static const char*source=R"MSL(#include <metal_stdlib>
using namespace metal;
typedef uint uint32_t;typedef int int32_t;typedef ulong uint64_t;typedef long int64_t;typedef uchar uint8_t;
struct Block {int x,y,z,r,side;};
struct Params {ulong base,ny,nz;uint count,nb;int xmin,ymin,zmin;uint items,threads;};
kernel void search(device const Block*blocks [[buffer(0)]],device uchar*flags [[buffer(1)]],constant Params&p [[buffer(2)]],uint tid [[thread_position_in_grid]]){
 for(uint item=0;item<p.items;item++){
 uint index=tid+item*p.threads;
 if(index>=p.count)break;
 ulong base=p.base,ny=p.ny,nz=p.nz;uint nb=p.nb;int xmin=p.xmin,ymin=p.ymin,zmin=p.zmin;

 uint64_t id=base+index;
 int32_t y=int32_t(int64_t(ymin)+int64_t(id%ny));id/=ny;
 int32_t z=int32_t(int64_t(zmin)+int64_t(id%nz));id/=nz;
 int32_t x=int32_t(int64_t(xmin)+int64_t(id));
 uint8_t ok=1;
 for(uint32_t i=0;i<nb;i++){
  Block b=blocks[i];
  int32_t bx=x+b.x,by=y+b.y,bz=z+b.z;
  // Java int multiplication wraps before sign extension to long.
  uint32_t xp=uint32_t(bx)*uint32_t(3129871);
  uint64_t sx=uint64_t(int64_t(int32_t(xp)));
  uint64_t l=sx^(uint64_t(int64_t(bz))*uint64_t(116129781))^uint64_t(int64_t(by));
  l=l*l*uint64_t(42317861)+l*uint64_t(11);
  // Arithmetic >>16 sign extension cannot affect the low 48 seed bits.
  uint64_t seed=(l>>16)^uint64_t(0x5DEECE66D);
  seed=(seed*uint64_t(0x5DEECE66D)+11)&uint64_t(0xFFFFFFFFFFFF);
  int r=int(seed>>46); // Java Random.nextInt(4), then side modulo 2.
  if(b.side)r%=2;
  if(r!=b.r){ok=0;break;}
 }
 flags[index]=ok;
 }

})MSL";

struct Tune {uint32_t width=128,items=1;};
static uint64_t hashBytes(uint64_t h,const void*p,size_t n){
 const auto*b=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;i++){h^=b[i];h*=1099511628211ULL;}return h;
}
// Timings include only the GPU interval where available. Correctness comparisons and
// command construction are outside that interval. Wall time is the older-OS fallback.
static double dispatch(id<MTLCommandQueue>queue,id<MTLComputePipelineState>pipe,
 id<MTLBuffer>bb,id<MTLBuffer>bf,Params p,Tune t){
 @autoreleasepool {
 p.items=t.items;p.threads=(p.count+t.items-1)/t.items;
 id<MTLCommandBuffer>cmd=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cmd computeCommandEncoder];
 if(!cmd||!enc)throw std::runtime_error("Metal encoder allocation failed");
 [enc setComputePipelineState:pipe];[enc setBuffer:bb offset:0 atIndex:0];[enc setBuffer:bf offset:0 atIndex:1];[enc setBytes:&p length:sizeof(p) atIndex:2];
 [enc dispatchThreads:MTLSizeMake(p.threads,1,1) threadsPerThreadgroup:MTLSizeMake(t.width,1,1)];
 [enc endEncoding];auto start=std::chrono::steady_clock::now();[cmd commit];[cmd waitUntilCompleted];
 if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cmd.error?[[cmd.error localizedDescription] UTF8String]:"Metal dispatch failed");
 double gpu=cmd.GPUEndTime-cmd.GPUStartTime;
 return gpu>0?gpu:std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 }
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
 h=hashBytes(h,o.dim,sizeof(o.dim));h=hashBytes(h,o.lo,sizeof(o.lo));
 NSString*key=[NSString stringWithFormat:@"v2|%@|%@|%@|%lu|%lu|%016llx",device.name,device.architecture.name,
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
 std::vector<std::vector<uint8_t>>expected;
 auto params=[&](uint64_t base){return Params{base,o.dim[1],o.dim[2],count,uint32_t(blocks.size()),o.lo[0],o.lo[1],o.lo[2],1,count};};
 for(auto base:bases){dispatch(queue,pipe,bb,bf,params(base),fallback);auto*f=static_cast<uint8_t*>(bf.contents);expected.emplace_back(f,f+count);}
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
   std::memset(bf.contents,0xa5,count); // Unwritten outputs cannot pass verification.
   seconds+=dispatch(queue,pipe,bb,bf,params(bases[j]),variants[v]);
   if(std::memcmp(bf.contents,expected[j].data(),count)!=0)throw std::runtime_error("Autotune variant correctness mismatch");
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
 const uint32_t batch=262144;
 id<MTLBuffer>bf=[device newBufferWithLength:batch options:MTLResourceStorageModeShared];
 if(!queue||!bf)throw std::runtime_error("Metal allocation failed");
 uint64_t done=0,matches=0;auto start=std::chrono::steady_clock::now();auto last=start;
 std::signal(SIGINT,stop_search);std::signal(SIGTERM,stop_search);
 for(int north:o.north){
  auto blocks=orient(original,north,o);
  id<MTLBuffer>bb=[device newBufferWithBytes:blocks.data() length:blocks.size()*sizeof(Block) options:MTLResourceStorageModeShared];if(!bb)throw std::runtime_error("Metal block allocation failed");
  Tune tune=autotune(device,queue,pipe,bb,bf,blocks,o,mode,path);
  for(uint64_t base=0;base<o.total&&!stopped;){
   uint32_t count=uint32_t(std::min<uint64_t>(batch,o.total-base));
   Params p{base,o.dim[1],o.dim[2],count,uint32_t(blocks.size()),o.lo[0],o.lo[1],o.lo[2],tune.items,0};
   dispatch(queue,pipe,bb,bf,p,tune);
   auto flags=static_cast<uint8_t*>(bf.contents);
   for(uint32_t i=0;i<count;i++)if(flags[i]){int32_t x,y,z;decode(base+i,o,x,y,z);std::cout<<x<<" "<<y<<" "<<z<<" "<<north<<"\n";matches++;}
   if(!std::cout)throw std::runtime_error("Output write failed");
   base+=count;done+=count;
   auto now=std::chrono::steady_clock::now();
   if(std::chrono::duration<double>(now-last).count()>=1){double sec=std::chrono::duration<double>(now-start).count();std::cerr<<"progress "<<done<<"/"<<o.total*o.north.size()<<" rate "<<double(done)/sec<<" candidates/s matches "<<matches<<"\n";last=now;}
  }
  if(stopped)break;
 }
 double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::cout.flush();
 std::cerr<<(stopped?"cancelled":"complete")<<" total "<<done<<" matches "<<matches<<" seconds "<<sec<<" rate "<<double(done)/std::max(sec,1e-9)<<" candidates/s\n";
 return stopped?130:0;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}}
