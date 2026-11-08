// Offline CPU evaluator of the repository's exact Java hash. No GPU or desktop jobs.
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <omp.h>
struct Block{int x,y,z,r,side;};
uint64_t xz(int x,int z){return uint64_t(int64_t(int32_t(uint32_t(x)*3129871u)))^(uint64_t(int64_t(z))*116129781ull);}
unsigned rot(int x,int y,int z){uint64_t l=xz(x,z)^uint64_t(int64_t(y));l=l*l*42317861ull+l*11ull;return unsigned(((((l>>16)^0x5deece66dull)*0x5deece66dull+11ull)>>46)&3);}
__m512i vr(uint64_t a,int y){
 __m512i ys=_mm512_set_epi64(int64_t(y+7),int64_t(y+6),int64_t(y+5),int64_t(y+4),int64_t(y+3),int64_t(y+2),int64_t(y+1),int64_t(y));
 auto l=_mm512_xor_si512(_mm512_set1_epi64(a),ys);
 l=_mm512_add_epi64(_mm512_mullo_epi64(_mm512_mullo_epi64(l,l),_mm512_set1_epi64(42317861)),_mm512_mullo_epi64(l,_mm512_set1_epi64(11)));
 auto s=_mm512_xor_si512(_mm512_srli_epi64(l,16),_mm512_set1_epi64(0x5deece66dull));
 return _mm512_and_si512(_mm512_srli_epi64(_mm512_add_epi64(_mm512_mullo_epi64(s,_mm512_set1_epi64(0x5deece66dull)),_mm512_set1_epi64(11)),46),_mm512_set1_epi64(3));
}
int main(int argc,char**argv){
 if(argc==5&&std::string(argv[1])=="--hash"){printf("%u\n",rot(atoi(argv[2]),atoi(argv[3]),atoi(argv[4])));return 0;}
 if(argc==5&&std::string(argv[1])=="--vector"){
  alignas(64) uint64_t values[8];_mm512_store_si512(values,vr(xz(atoi(argv[2]),atoi(argv[4])),atoi(argv[3])));
  for(auto v:values)printf("%llu ",(unsigned long long)v);puts("");return 0;
 }
 if(argc!=9){fprintf(stderr,"formation xmin xmax ymin ymax zmin zmax threads\n");return 2;}
 std::ifstream f(argv[1]);std::string line;
 if(!std::getline(f,line))return 3;int count=std::stoi(line);std::vector<Block> blocks;
 while(std::getline(f,line)){std::istringstream s(line);Block b;if(s>>b.x>>b.y>>b.z>>b.r>>b.side)blocks.push_back(b);}
 if(count<1||count>512||int(blocks.size())!=count)return 4;
 int xmin=atoi(argv[2]),xmax=atoi(argv[3]),ymin=atoi(argv[4]),ymax=atoi(argv[5]),zmin=atoi(argv[6]),zmax=atoi(argv[7]);
 if(xmin>xmax||ymin>ymax||zmin>zmax)return 5;
 omp_set_num_threads(atoi(argv[8])); auto start=std::chrono::steady_clock::now();
 uint64_t tested=0,matches=0;
 for(int n=0;n<4;n++){
  auto bs=blocks;for(auto &b:bs){for(int j=0;j<n;j++){int x=b.x;b.x=-b.z;b.z=x;}b.r=(b.r+n)%(b.side?2:4);}
  #pragma omp parallel for schedule(dynamic,4) reduction(+:matches)
  for(int x=xmin;x<=xmax;x++){
   for(int z=zmin;z<=zmax;z++){
    uint64_t first[4];int pre=std::min(4,count);
    for(int j=0;j<pre;j++)first[j]=xz(x+bs[j].x,z+bs[j].z);
    for(int y=ymin;y<=ymax;y+=8){
     __mmask8 good=(y+7<=ymax)?255:((1u<<(ymax-y+1))-1);
     for(int j=0;j<pre&&good;j++){
      auto v=vr(first[j],y+bs[j].y);if(bs[j].side)v=_mm512_and_si512(v,_mm512_set1_epi64(1));
      good&=_mm512_cmpeq_epi64_mask(v,_mm512_set1_epi64(bs[j].r));
     }
     while(good){
      int lane=__builtin_ctz((unsigned)good);good&=good-1;bool ok=true;
      for(int j=pre;j<count;j++){auto &b=bs[j];if((rot(x+b.x,y+lane+b.y,z+b.z)&(b.side?1:3))!=unsigned(b.r)){ok=false;break;}}
      if(ok){matches++;
       #pragma omp critical
       {printf("%d %d %d %d\n",x,y+lane,z,n);fflush(stdout);}
      }
     }
    }
   }
  }
  tested+=uint64_t(xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1);
  fprintf(stderr,"north %d complete, tested %llu, seconds %.3f\n",n,(unsigned long long)tested,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());fflush(stderr);
 }
 fprintf(stderr,"COMPLETE tested=%llu matches=%llu seconds=%.3f\n",(unsigned long long)tested,(unsigned long long)matches,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
}
