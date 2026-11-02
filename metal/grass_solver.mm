#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "solver_common.h"
struct Params {uint64_t base,ny,nz;uint32_t count,nb;int32_t xmin,ymin,zmin;};
static const char*source=R"MSL(#include <metal_stdlib>
using namespace metal;
typedef uint uint32_t;typedef int int32_t;typedef ulong uint64_t;typedef long int64_t;typedef uchar uint8_t;
struct Block {int x,y,z,r,side;};
struct Params {ulong base,ny,nz;uint count,nb;int xmin,ymin,zmin;};
kernel void search(device const Block*blocks [[buffer(0)]],device uchar*flags [[buffer(1)]],constant Params&p [[buffer(2)]],uint tid [[thread_position_in_grid]]){
 if(tid>=p.count)return;
 ulong base=p.base,ny=p.ny,nz=p.nz;uint nb=p.nb;int xmin=p.xmin,ymin=p.ymin,zmin=p.zmin;

 uint64_t id=base+tid;
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
 flags[tid]=ok;

})MSL";
int main(int argc,char**argv){@autoreleasepool{try{
 auto o=options(argc,argv);auto original=load(o.path);
 for(int n:o.north)orient(original,n,o);
 id<MTLDevice>device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("No Metal device");
 NSError*error=nil;
 id<MTLLibrary>lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&error];
 if(!lib)throw std::runtime_error([[error localizedDescription] UTF8String]);
 id<MTLComputePipelineState>pipe=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"search"] error:&error];
 if(!pipe)throw std::runtime_error([[error localizedDescription] UTF8String]);
 id<MTLCommandQueue>queue=[device newCommandQueue];
 id<MTLBuffer>bf=[device newBufferWithLength:262144 options:MTLResourceStorageModeShared];
 if(!queue||!bf)throw std::runtime_error("Metal allocation failed");
 
 const uint32_t batch=262144;
 uint64_t done=0,matches=0;auto start=std::chrono::steady_clock::now();auto last=start;
 std::signal(SIGINT,stop_search);std::signal(SIGTERM,stop_search);
 for(int north:o.north){
  auto blocks=orient(original,north,o);
  id<MTLBuffer>bb=[device newBufferWithBytes:blocks.data() length:blocks.size()*sizeof(Block) options:MTLResourceStorageModeShared];if(!bb)throw std::runtime_error("Metal block allocation failed");
  for(uint64_t base=0;base<o.total&&!stopped;){
   uint32_t count=uint32_t(std::min<uint64_t>(batch,o.total-base));
   @autoreleasepool {
   Params p{base,o.dim[1],o.dim[2],count,uint32_t(blocks.size()),o.lo[0],o.lo[1],o.lo[2]};
   id<MTLCommandBuffer>cmd=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cmd computeCommandEncoder];
   if(!cmd||!enc)throw std::runtime_error("Metal encoder allocation failed");
   [enc setComputePipelineState:pipe];[enc setBuffer:bb offset:0 atIndex:0];[enc setBuffer:bf offset:0 atIndex:1];[enc setBytes:&p length:sizeof(p) atIndex:2];
   NSUInteger width=std::min<NSUInteger>(256,pipe.maxTotalThreadsPerThreadgroup);
   [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
   [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
   if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cmd.error?[[cmd.error localizedDescription] UTF8String]:"Metal dispatch failed");
   }
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
