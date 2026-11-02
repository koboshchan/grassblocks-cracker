#include "solver_common.h"
#include <cuda_runtime.h>
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void search(const Block*blocks,uint32_t nb,uint8_t*flags,uint64_t base,uint32_t count,int32_t xmin,int32_t ymin,int32_t zmin,uint64_t ny,uint64_t nz){
 uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=count)return;

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

}
int main(int argc,char**argv){try{
 auto o=options(argc,argv);auto original=load(o.path);
 // Validate every north before creating a device context or printing partial results.
 for(int n:o.north)orient(original,n,o);
 uint8_t *df=nullptr;Block *db=nullptr;std::vector<uint8_t> host(262144);
 check(cudaMalloc((void**)&df,host.size()));check(cudaMalloc((void**)&db,original.size()*sizeof(Block)));

 const uint32_t batch=262144;
 uint64_t done=0,matches=0;auto start=std::chrono::steady_clock::now();auto last=start;
 std::signal(SIGINT,stop_search);std::signal(SIGTERM,stop_search);
 for(int north:o.north){
  auto blocks=orient(original,north,o);
  check(cudaMemcpy(db,blocks.data(),blocks.size()*sizeof(Block),cudaMemcpyHostToDevice));
  for(uint64_t base=0;base<o.total&&!stopped;){
   uint32_t count=uint32_t(std::min<uint64_t>(batch,o.total-base));
   search<<<(count+255)/256,256>>>(db,uint32_t(blocks.size()),df,base,count,o.lo[0],o.lo[1],o.lo[2],o.dim[1],o.dim[2]);
   check(cudaGetLastError());check(cudaMemcpy(host.data(),df,count,cudaMemcpyDeviceToHost));auto flags=host.data();
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

}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
