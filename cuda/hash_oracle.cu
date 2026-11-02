#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#define main solver_main
#include "grass_solver.cu"
#undef main
struct Query {int32_t x,y,z,side;};
__global__ void hash_queries(const Query*q,uint8_t*out,uint32_t n){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=texture(q[i].x,q[i].y,q[i].z,q[i].side);}
int main(){try{
#ifdef _WIN32
_setmode(_fileno(stdin),_O_BINARY);
#endif
uint32_t n;std::cin.read((char*)&n,4);if(!std::cin||n>1000000)return 1;std::vector<Query> q(n);std::cin.read((char*)q.data(),size_t(n)*sizeof(Query));if(!std::cin)return 1;
Query*d;uint8_t*r;check(cudaMalloc((void**)&d,size_t(n)*sizeof(Query)));check(cudaMalloc((void**)&r,n));check(cudaMemcpy(d,q.data(),size_t(n)*sizeof(Query),cudaMemcpyHostToDevice));
hash_queries<<<(n+255)/256,256>>>(d,r,n);check(cudaGetLastError());std::vector<uint8_t>out(n);check(cudaMemcpy(out.data(),r,n,cudaMemcpyDeviceToHost));
#ifdef _WIN32
_setmode(_fileno(stdout),_O_BINARY);
#endif
std::cout.write((char*)out.data(),n);return 0;}catch(const std::exception&e){std::cerr<<e.what();return 1;}}
