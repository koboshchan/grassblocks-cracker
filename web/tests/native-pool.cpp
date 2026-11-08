// Small, deterministic protocol tests. No image QA or browser worker is launched.
#include "../wasm/pipeline.cpp"
#include <cassert>
#include <iostream>
#include <limits>
namespace {
int failRatio=0,siftCalls=0,binaryCalls=0,emptyCalls=0;
bool binaryThrows=false;
std::vector<int> unionOrder;
}
namespace grass {
double clockMs(){return 0;}
std::vector<Quad>siftSeeds(const std::vector<float>&,int,int,const std::array<double,256>&,double ratio,FeatureTimings*){
 ++siftCalls;if(int(std::round(ratio*100))==failRatio)throw std::runtime_error("injected sift failure");
 if(failRatio==-1)return {};
 Quad q{};q[0].x=ratio;q[1].x=std::nextafter(ratio,1.);return {q};
}
std::vector<Quad>binarySeeds(const std::vector<float>&,int,int,const std::array<double,256>&,FeatureTimings*){
 ++binaryCalls;if(binaryThrows)throw std::runtime_error("injected binary failure");return {};
}
std::vector<Verdict>classifyBatchNative(const Context&,const std::vector<Quad>&){return {{}};}
Result runDetection(Context&c,const std::vector<Quad>&raw,bool manual,Timings&){
 assert(!manual); // Empty automatic proposals MUST keep topDown enabled.
 if(raw.empty())++emptyCalls;
 Result r;Block b;b.valid=true;b.x=int(std::round(c.opts.siftRatio*100));b.y=int(c.opts.refitPlane);b.z=int(c.opts.refitFlat);
 b.score=std::nextafter(.88,1.);b.margin=std::nextafter(.35,0.);b.quad=raw.empty()?Quad{}:raw[0];b.rotation=2;b.votes=4;
 r.blocks={b};r.seeds={b};r.reason="trial-zero reason";r.mode="same metadata";r.camera[0]=std::nextafter(12.,13.);r.planeH[0]=1.;r.hasCamera=r.hasPlane=true;return r;
}
std::vector<Block>mergeAligned(const std::vector<Block>&a,const std::vector<Block>&b){unionOrder.push_back(b[0].x);auto out=a;out.insert(out.end(),b.begin(),b.end());return out;}
}
int main(){
 unsigned char image[8*8*4]{};double ref[256]{};
 auto run=[&](int failure,bool binaryError,bool allMissing){
  failRatio=failure;binaryThrows=binaryError;siftFailed=false;siftCalls=binaryCalls=emptyCalls=0;unionOrder.clear();
  assert(grass_detect(image,8,8,ref,nullptr,nullptr,0,1)==0);std::string expected=output;
  int serialSift=siftCalls,serialBinary=binaryCalls,serialEmpty=emptyCalls;auto serialOrder=unionOrder;
  siftFailed=false;siftCalls=binaryCalls=emptyCalls=0;unionOrder.clear();
  assert(grass_pool_prepare(image,8,8,ref)==0);assert(siftCalls==serialSift);assert(binaryCalls==serialBinary);
  auto proposals=pool.raw;auto cfg=configuration(true);
  if(!allMissing){for(int i=8;i>=0;--i){std::vector<double>raw;for(auto q:proposals[i])for(auto p:q){raw.push_back(p.x);raw.push_back(p.y);}
   assert(grass_pool_trial(image,8,8,ref,i,raw.data(),proposals[i].size())==0);
   auto opaque=blob;assert(grass_pool_submit(i,opaque.data(),opaque.size())==0);
   assert(grass_pool_submit(i,opaque.data(),opaque.size())==-1); // Duplicate rejection does not poison state.
  }}
  assert(grass_pool_finish()==0);assert(output==expected);assert(emptyCalls==serialEmpty);assert(unionOrder==serialOrder);
  assert(output.find("\"camera\":null,\"planeH\":null")!=std::string::npos);
  assert(output.find("0.88000000000000012")!=std::string::npos);
  if(failure==76){assert(siftFailed);assert(siftCalls==2);assert(binaryCalls==3);}
 };
 run(0,false,false);run(-1,false,false);run(76,false,false);run(76,true,false);run(76,true,true);
 // Sticky failure survives the next request, with binary retried per trial.
 siftCalls=binaryCalls=0;assert(grass_pool_prepare(image,8,8,ref)==0);assert(siftCalls==0);assert(binaryCalls==9);
 unsigned char broken[4]{};assert(grass_pool_submit(0,broken,4)==-1);assert(grass_pool_finish()==0);
 // Snapshot bit patterns, not decimal or float32 transport; checked length guards.
 grass::Result r;r.reason="a\"b\\c";grass::Block b;b.valid=true;b.score=-0.;b.margin=std::nextafter(.1,1.);b.quad[0].x=std::nextafter(123.456,124.);r.blocks={b};grass::Timings t;snapshot(r,t);
 grass::Result recovered;grass::Timings rt;readSnapshot(blob.data(),blob.size(),recovered,rt);
 assert(std::signbit(recovered.blocks[0].score));assert(std::memcmp(&b.margin,&recovered.blocks[0].margin,8)==0);assert(std::memcmp(&b.quad[0].x,&recovered.blocks[0].quad[0].x,8)==0);assert(r.reason==recovered.reason);
 bool rejected=false;try{readSnapshot(blob.data(),blob.size()-1,recovered,rt);}catch(...){rejected=true;}assert(rejected);
 grass_pool_abort();assert(!pool.active);std::cout<<"Native pool protocol PASS (fixed order, metadata, binary64, empty automatic, cache, sticky failure, serial fallback, malformed transport)\n";
}
