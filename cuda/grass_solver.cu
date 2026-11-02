#include "solver_common.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
__constant__ Block cached_blocks[1024];
static uint32_t tuning(const char*name,uint32_t def,uint32_t lo,uint32_t hi){const char*s=std::getenv(name);if(!s)return def;auto v=number(s);if(v<lo||v>hi)throw std::runtime_error(std::string(name)+" outside supported range");return uint32_t(v);}
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ __forceinline__ int texture(int32_t bx,int32_t by,int32_t bz,int side){
  // Java int multiplication wraps before sign extension to long.
  uint32_t xp=uint32_t(bx)*uint32_t(3129871);
  uint64_t sx=uint64_t(int64_t(int32_t(xp)));
  uint64_t l=sx^(uint64_t(int64_t(bz))*uint64_t(116129781))^uint64_t(int64_t(by));
  l=l*l*uint64_t(42317861)+l*uint64_t(11);
  // Arithmetic >>16 sign extension cannot affect the low 48 seed bits.
  uint64_t seed=(l>>16)^uint64_t(0x5DEECE66D);
  seed=(seed*uint64_t(0x5DEECE66D)+11)&uint64_t(0xFFFFFFFFFFFF);
  int r=int(seed>>46); // Java Random.nextInt(4), then side modulo 2.
  if(side)r%=2;

 return r;
}
__global__ void search(const Block*blocks,uint32_t nb,uint32_t*results,uint32_t*result_count,uint64_t base,uint32_t count,uint32_t items,int32_t xmin,int32_t ymin,int32_t zmin,uint64_t ny,uint64_t nz){
 uint32_t first=(blockIdx.x*blockDim.x+threadIdx.x)*items;
 for(uint32_t j=0;j<items;j++){
 uint32_t tid=first+j;if(tid>=count)break;

 uint64_t id=base+tid;
 int32_t y=int32_t(int64_t(ymin)+int64_t(id%ny));id/=ny;
 int32_t z=int32_t(int64_t(zmin)+int64_t(id%nz));id/=nz;
 int32_t x=int32_t(int64_t(xmin)+int64_t(id));
 uint8_t ok=1;
 for(uint32_t i=0;i<nb;i++){
  Block b=nb<=1024?cached_blocks[i]:blocks[i];
  int32_t bx=x+b.x,by=y+b.y,bz=z+b.z;
  int r=texture(bx,by,bz,b.side);
  if(r!=b.r){ok=0;break;}
 }
 if(ok)results[atomicAdd(result_count,1u)]=tid;
 }

}
int main(int argc,char**argv){try{
 std::string mode="auto";std::vector<char*> args{argv[0]};
 for(int i=1;i<argc;i++){if(std::string(argv[i])=="--autotune"){if(++i==argc)throw std::runtime_error("Missing autotune mode");mode=argv[i];if(mode!="auto"&&mode!="force"&&mode!="off")throw std::runtime_error("autotune expects auto|force|off");}else args.push_back(argv[i]);}
 auto o=options(int(args.size()),args.data());auto original=load(o.path);
 // Validate every north before creating a device context or printing partial results.
 for(int n:o.north)orient(original,n,o);
 const uint32_t batch=tuning("GRASS_BATCH",4194304,256,16777216);
 uint32_t threads=tuning("GRASS_THREADS",128,32,1024);
 uint32_t items=tuning("GRASS_ITEMS",1,1,8);
 if(threads%32)throw std::runtime_error("GRASS_THREADS must be multiple of 32");
 uint32_t *df=nullptr,*dc=nullptr;Block *db=nullptr;std::vector<uint32_t> host;
 check(cudaMalloc((void**)&df,size_t(batch)*sizeof(uint32_t)));check(cudaMalloc((void**)&dc,sizeof(uint32_t)));check(cudaMalloc((void**)&db,original.size()*sizeof(Block)));



 cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
 auto tuneBlocks=orient(original,o.north[0],o);
 std::stable_sort(tuneBlocks.begin(),tuneBlocks.end(),[](const Block&a,const Block&b){return a.side<b.side;});
 check(cudaMemcpy(db,tuneBlocks.data(),tuneBlocks.size()*sizeof(Block),cudaMemcpyHostToDevice));
 if(tuneBlocks.size()<=1024)check(cudaMemcpyToSymbol(cached_blocks,tuneBlocks.data(),tuneBlocks.size()*sizeof(Block)));
 // Cache includes GPU, architecture, kernel version, problem dimensions and exact formation.
 std::string key=std::string(prop.name)+"/"+std::to_string(prop.major)+"."+std::to_string(prop.minor)+"/sparse-v3/"+std::to_string(o.dim[1])+"/"+std::to_string(o.dim[2]);
 for(auto b:tuneBlocks)key+="/"+std::to_string(b.x)+","+std::to_string(b.y)+","+std::to_string(b.z)+","+std::to_string(b.r)+","+std::to_string(b.side);
 uint64_t hash=1469598103934665603ull;for(unsigned char c:key){hash^=c;hash*=1099511628211ull;}
 const char*root=std::getenv("LOCALAPPDATA");if(!root)root=std::getenv("HOME");
 std::filesystem::path cache;
 if(root){cache=std::filesystem::path(root)/".grass-solvers"/("cuda-"+std::to_string(hash)+".txt");}
 bool manual=std::getenv("GRASS_THREADS")||std::getenv("GRASS_ITEMS");
 bool cached=false;
 if(mode=="auto"&&!manual&&!cache.empty()){
  std::ifstream f(cache);std::string saved;uint32_t t=0,it=0;if(std::getline(f,saved)&&saved==key&&(f>>t>>it)&&
   (t==64||t==128||t==256||t==512)&&(it==1||it==2||it==4)){threads=t;items=it;cached=true;
   std::cerr<<"autotune cache hit "<<threads<<" threads "<<items<<" items\n";}
 }
 if(mode!="off"&&!manual&&!cached){
  uint32_t n=uint32_t(std::min<uint64_t>(batch,std::min<uint64_t>(o.total,4194304)));
  // Fixed candidate subset for every configuration, bit-exact sorted-result agreement.
  std::vector<uint32_t> reference;
  float best=std::numeric_limits<float>::max();
  cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
  for(uint32_t t:{64u,128u,256u,512u})for(uint32_t it:{1u,2u,4u}){
   auto launch=[&](){check(cudaMemset(dc,0,sizeof(uint32_t)));
    search<<<(n+t*it-1)/(t*it),t>>>(db,uint32_t(tuneBlocks.size()),df,dc,0,n,it,o.lo[0],o.lo[1],o.lo[2],o.dim[1],o.dim[2]);check(cudaGetLastError());};
   launch();check(cudaDeviceSynchronize());
   check(cudaEventRecord(begin));
   for(int rep=0;rep<3;rep++)launch();
   check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms=0;check(cudaEventElapsedTime(&ms,begin,end));ms/=3;
   uint32_t found=0;check(cudaMemcpy(&found,dc,sizeof(found),cudaMemcpyDeviceToHost));if(found>n)throw std::runtime_error("Autotune count overflow");
   std::vector<uint32_t> ids(found);if(found)check(cudaMemcpy(ids.data(),df,size_t(found)*4,cudaMemcpyDeviceToHost));std::sort(ids.begin(),ids.end());
   if(t==64&&it==1)reference=ids;else if(ids!=reference)throw std::runtime_error("Autotune correctness mismatch");
   std::cerr<<"autotune trial threads "<<t<<" items "<<it<<" kernel_ms "<<ms<<" origins/s "<<double(n)/(ms*.001)<<"\n";
   if(ms<best){best=ms;threads=t;items=it;}
  }
  check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
  std::cerr<<"autotune selected "<<threads<<" threads "<<items<<" items; exact outputs agree\n";
  if(!cache.empty()){std::error_code ec;std::filesystem::create_directories(cache.parent_path(),ec);if(!ec){std::ofstream f(cache);if(f)f<<key<<"\n"<<threads<<" "<<items<<"\n";}}
 }
 std::cerr<<"CUDA "<<prop.name<<" sm_"<<prop.major<<prop.minor<<" threads "<<threads<<" items "<<items<<" batch "<<batch<<"\n";

 uint64_t done=0,matches=0;auto start=std::chrono::steady_clock::now();auto last=start;
 std::signal(SIGINT,stop_search);std::signal(SIGTERM,stop_search);
 for(int north:o.north){
  auto blocks=orient(original,north,o);
  // Tops reject 75% versus sides 50%; stable ordering preserves equal-information rows.
  std::stable_sort(blocks.begin(),blocks.end(),[](const Block&a,const Block&b){return a.side<b.side;});
  if(blocks.size()<=1024)check(cudaMemcpyToSymbol(cached_blocks,blocks.data(),blocks.size()*sizeof(Block)));
  check(cudaMemcpy(db,blocks.data(),blocks.size()*sizeof(Block),cudaMemcpyHostToDevice));
  for(uint64_t base=0;base<o.total&&!stopped;){
   uint32_t count=uint32_t(std::min<uint64_t>(batch,o.total-base));
   check(cudaMemset(dc,0,sizeof(uint32_t)));
   search<<<(count+threads*items-1)/(threads*items),threads>>>(db,uint32_t(blocks.size()),df,dc,base,count,items,o.lo[0],o.lo[1],o.lo[2],o.dim[1],o.dim[2]);
   check(cudaGetLastError());uint32_t found=0;check(cudaMemcpy(&found,dc,sizeof(found),cudaMemcpyDeviceToHost));
   // Capacity is count: at most one index per origin. No capped buffer or lost matches.
   if(found>count)throw std::runtime_error("Invalid GPU result count");
   host.resize(found);if(found)check(cudaMemcpy(host.data(),df,size_t(found)*sizeof(uint32_t),cudaMemcpyDeviceToHost));
   std::sort(host.begin(),host.end());
   for(uint32_t i:host){int32_t x,y,z;decode(base+i,o,x,y,z);std::cout<<x<<" "<<y<<" "<<z<<" "<<north<<"\n";matches++;}
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
