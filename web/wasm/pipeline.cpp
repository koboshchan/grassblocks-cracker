// End-to-end detector ABI. No JavaScript callbacks execute algorithm stages.
#include "geometry.h"
#include <emscripten/emscripten.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <map>
#include <stdexcept>
#include <cstring>
// Observability only; callbacks never execute algorithm stages or change results.
EM_JS(void, grass_progress, (int completed, int total, const char* name, const char* stage), {
 if (Module['onDetectionProgress']) {
  try { Module['onDetectionProgress'](completed, total, UTF8ToString(name), UTF8ToString(stage)); } catch (_) {}
 }
});
namespace {
using namespace grass;
std::string output;
bool siftFailed=false;
void number(std::ostream&o,double v){if(std::isfinite(v))o<<std::setprecision(17)<<v;else o<<"null";}
void text(std::ostream&o,const std::string&s){o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32){const char*h="0123456789abcdef";o<<"\\u00"<<h[c>>4]<<h[c&15];}else o<<c;}o<<'"';}
void verdict(std::ostream&o,const Verdict&b){o<<"\"rotation\":"<<b.rotation<<",\"score\":";number(o,b.score);o<<",\"margin\":";number(o,b.margin);o<<",\"votes\":"<<b.votes;}
void block(std::ostream&o,const Block&b){o<<'{';verdict(o,b);o<<",\"quad\":[";for(int i=0;i<4;i++){if(i)o<<',';o<<'[';number(o,b.quad[i].x);o<<',';number(o,b.quad[i].y);o<<']';}o<<"],\"x\":"<<b.x<<",\"y\":"<<b.y<<",\"z\":"<<b.z<<'}';}
void blocks(std::ostream&o,const std::vector<Block>&bs){o<<'[';for(size_t i=0;i<bs.size();i++){if(i)o<<',';block(o,bs[i]);}o<<']';}
void timing(std::ostream&o,const Timings&t){o<<"{\"green\":";number(o,t.green);o<<",\"features\":";number(o,t.features);o<<",\"featureExtraction\":";number(o,t.featureExtraction);o<<",\"matchingHomography\":";number(o,t.matchingHomography);o<<",\"ecc\":";number(o,t.ecc);o<<",\"latticeFallback\":";number(o,t.latticeFallback);o<<",\"trialUnion\":";number(o,t.trialUnion);o<<",\"verify\":";number(o,t.verify);o<<",\"plane\":";number(o,t.plane);o<<",\"voxel\":";number(o,t.voxel);o<<",\"total\":";number(o,t.total);o<<'}';}
Context context(const unsigned char*rgba,int w,int h,const double*ref){if(!rgba||!ref||w<8||h<8||double(w)*h>16*1024*1024)throw std::runtime_error("Invalid native image/reference");Context c;c.w=w;c.h=h;c.g.resize(size_t(w)*h);for(size_t i=0;i<c.g.size();i++)c.g[i]=rgba[4*i+1];for(int i=0;i<256;i++){if(!std::isfinite(ref[i])||ref[i]<0||ref[i]>255)throw std::runtime_error("Invalid reference pixel");c.ref[i]=ref[i];}for(int k=0;k<4;k++)for(int y=0;y<16;y++)for(int x=0;x<16;x++){int xx=x,yy=y;for(int j=0;j<k;j++){int t=xx;xx=yy;yy=15-t;}c.refs[k][y*16+x]=c.ref[yy*16+xx];}return c;}
Options options(const double*p){Options o;if(!p)return o;for(int i=0;i<12;i++)if(!std::isfinite(p[i]))throw std::runtime_error("Nonfinite native option");o.accept=p[0];o.margin=p[1];o.relaxed=p[2];o.coarse=p[3];if(int(p[11])&1)o.refineMin=p[4];if(int(p[11])&2)o.refineMargin=p[5];o.refineShift=p[6];o.siftRatio=p[7];o.refineVotes=p[8];o.refitPlane=p[9]!=0;o.refitFlat=p[10]!=0;return o;}
Quad canonical(Quad q){int start=0;for(int i=0;i<4;i++)if(q[i].x+q[i].y<q[start].x+q[start].y)start=i;Quad out;for(int i=0;i<4;i++)out[i]=q[(start+i)%4];double area=0;for(int i=0;i<4;i++)area=(area+out[i].x*out[(i+1)%4].y)-out[(i+1)%4].x*out[i].y;if(area<0)std::swap(out[1],out[3]);return out;}
void error(const std::exception&e){std::ostringstream o;o<<"{\"error\":";text(o,e.what());o<<'}';output=o.str();}
struct Trial {std::string name;int tiles,mergedTiles;Timings t;};

std::vector<std::pair<std::string,Options>>configuration(bool automatic,Options base={}){std::vector<std::pair<std::string,Options>>config;Options o;if(automatic){config.push_back({"original",o});o.accept=.84;o.relaxed=.72;o.margin=.30;config.push_back({"relaxed",o});o=Options{};o.refineMin=.62;o.refineMargin=.20;config.push_back({"refine",o});o.refitPlane=true;config.push_back({"refit-refine",o});o=Options{};o.siftRatio=.76;config.push_back({"ratio",o});o=Options{};o.accept=.84;o.relaxed=.72;o.margin=.30;o.refineMin=.62;o.refineMargin=.20;o.refitPlane=true;config.push_back({"combo",o});o=Options{};o.refitFlat=true;o.siftRatio=.76;config.push_back({"flat-ratio",o});o=Options{};o.refitFlat=true;o.refineMin=.55;o.refineMargin=.10;o.refineVotes=2;o.refineShift=6;config.push_back({"rescue",o});o=Options{};o.siftRatio=.80;config.push_back({"ratio80",o});}else config.push_back({"manual",base});return config;}
std::string serialize(const Result&result,std::vector<Trial>&trials,const Timings&total,bool multiple){std::ostringstream out;out<<"{\"blocks\":";blocks(out,result.blocks);out<<",\"seeds\":";blocks(out,result.seeds);out<<",\"reason\":";text(out,result.reason);out<<",\"mode\":";text(out,result.mode);out<<",\"camera\":";if(result.hasCamera){out<<'[';for(int i=0;i<12;i++){if(i)out<<',';number(out,result.camera[i]);}out<<']';}else out<<"null";out<<",\"planeH\":";if(result.hasPlane){out<<'[';for(int i=0;i<9;i++){if(i)out<<',';number(out,result.planeH[i]);}out<<']';}else out<<"null";out<<",\"notes\":[\"Native end-to-end WASM pipeline; no JavaScript algorithm fallback.\"],\"trial\":";text(out,multiple?"original-anchored geometric union":"single pass");out<<",\"trials\":[";for(size_t i=0;i<trials.size();i++){if(i)out<<',';auto&t=trials[i];out<<"{\"name\":";text(out,t.name);out<<",\"tiles\":"<<t.tiles<<",\"mergedTiles\":"<<t.mergedTiles<<",\"seconds\":";number(out,t.t.total/1000);out<<",\"timingsMs\":";timing(out,t.t);out<<",\"notes\":[]}";}out<<"],\"timingsMs\":";timing(out,total);out<<",\"seconds\":";number(out,total.total/1000);out<<",\"backend\":\"WASM native pipeline (OpenCV + VLFeat + SIMD)\"}";return out.str();}
struct PoolState {
 Context c; std::vector<std::vector<Quad>> raw; std::vector<Timings> timings;
 std::vector<Result> candidates; std::vector<bool> received; double started=0; Timings total;
 bool active=false;
} pool;
std::vector<unsigned char> blob;
// Same-build, ephemeral worker transport. Encode fields individually, never copy
// C++ objects/padding or round floats to float32. Lengths are checked on import.
struct Writer {
 std::vector<unsigned char> data;
 template<class T>void scalar(const T&v){auto p=reinterpret_cast<const unsigned char*>(&v);data.insert(data.end(),p,p+sizeof(v));}
 void str(const std::string&v){scalar(uint32_t(v.size()));data.insert(data.end(),v.begin(),v.end());}
 void list(const std::vector<Block>&bs){scalar(uint32_t(bs.size()));for(auto&b:bs){scalar(int32_t(b.valid));scalar(int32_t(b.rotation));scalar(int32_t(b.votes));scalar(b.score);scalar(b.margin);for(auto p:b.quad){scalar(p.x);scalar(p.y);}scalar(int32_t(b.x));scalar(int32_t(b.y));scalar(int32_t(b.z));}}
};
struct Reader {
 const unsigned char*p;size_t left;
 template<class T>T scalar(){if(left<sizeof(T))throw std::runtime_error("Truncated trial snapshot");T v;std::memcpy(&v,p,sizeof(v));p+=sizeof(v);left-=sizeof(v);return v;}
 uint32_t count(uint32_t max){auto n=scalar<uint32_t>();if(n>max)throw std::runtime_error("Oversized trial snapshot");return n;}
 std::string str(){auto n=count(65536);if(left<n)throw std::runtime_error("Truncated trial text");std::string v(reinterpret_cast<const char*>(p),n);p+=n;left-=n;return v;}
 std::vector<Block>list(){std::vector<Block>out;auto n=count(100000);if(size_t(n)*104>left)throw std::runtime_error("Truncated trial blocks");for(uint32_t i=0;i<n;i++){Block b;b.valid=scalar<int32_t>()!=0;b.rotation=scalar<int32_t>();b.votes=scalar<int32_t>();b.score=scalar<double>();b.margin=scalar<double>();for(auto&v:b.quad){v.x=scalar<double>();v.y=scalar<double>();}b.x=scalar<int32_t>();b.y=scalar<int32_t>();b.z=scalar<int32_t>();out.push_back(b);}return out;}
};
void snapshot(const Result&r,const Timings&t){Writer w;w.scalar(uint32_t(0x47525031));w.list(r.blocks);w.list(r.seeds);w.str(r.reason);w.str(r.mode);for(double v:r.camera)w.scalar(v);for(double v:r.planeH)w.scalar(v);w.scalar(int32_t(r.hasCamera));w.scalar(int32_t(r.hasPlane));for(double v:{t.ecc,t.verify,t.plane,t.voxel,t.latticeFallback,t.total})w.scalar(v);blob=std::move(w.data);}
void readSnapshot(const unsigned char*p,int size,Result&r,Timings&t){if(!p||size<0||size>32*1024*1024)throw std::runtime_error("Invalid trial snapshot");Reader in{p,size_t(size)};if(in.scalar<uint32_t>()!=0x47525031)throw std::runtime_error("Wrong trial snapshot version");r.blocks=in.list();r.seeds=in.list();r.reason=in.str();r.mode=in.str();for(double&v:r.camera)v=in.scalar<double>();for(double&v:r.planeH)v=in.scalar<double>();r.hasCamera=in.scalar<int32_t>()!=0;r.hasPlane=in.scalar<int32_t>()!=0;t.ecc=in.scalar<double>();t.verify=in.scalar<double>();t.plane=in.scalar<double>();t.voxel=in.scalar<double>();t.latticeFallback=in.scalar<double>();t.total=in.scalar<double>();if(in.left)throw std::runtime_error("Trailing trial snapshot data");}
void requirePool(){if(!pool.active)throw std::runtime_error("No prepared automatic detection");}

}
extern "C" {
EMSCRIPTEN_KEEPALIVE const char*grass_result(){return output.c_str();}
EMSCRIPTEN_KEEPALIVE int grass_classify(const unsigned char*rgba,int w,int h,const double*ref,const double*quad){try{Context c=context(rgba,w,h,ref);if(!quad)throw std::runtime_error("Missing quad");Quad q;for(int i=0;i<4;i++){if(!std::isfinite(quad[2*i])||!std::isfinite(quad[2*i+1]))throw std::runtime_error("Nonfinite quad");q[i]={quad[2*i],quad[2*i+1]};}auto r=classifyBatchNative(c,{q})[0];std::ostringstream o;o<<"{\"ok\":"<<(r.valid?"true":"false")<<',';verdict(o,r);o<<'}';output=o.str();return 0;}catch(const std::exception&e){error(e);return -1;}}
// Feature proposals only. Verification, plane growth and voxel classification stay on GPU.
EMSCRIPTEN_KEEPALIVE int grass_feature_seeds(const unsigned char*rgba,int w,int h,const double*ref,const double*opts){
 try{
  double start=clockMs();FeatureTimings timings;
  Context c=context(rgba,w,h,ref);c.opts=options(opts);
  std::vector<float>gf(c.g.begin(),c.g.end());
  std::vector<Quad>seeds;const char*backend="SIFT";
  try{
   if(siftFailed)throw std::runtime_error("SIFT previously failed");
   seeds=siftSeeds(gf,c.w,c.h,c.ref,c.opts.siftRatio,&timings);
  }catch(const std::exception&){
   backend="binary";seeds=binarySeeds(gf,c.w,c.h,c.ref,&timings);
  }
  std::ostringstream out;out<<"{\"seeds\":[";
  for(size_t i=0;i<seeds.size();i++){
   if(i)out<<',';out<<'[';
   for(int k=0;k<4;k++){
    if(k)out<<',';out<<'[';number(out,seeds[i][k].x);out<<',';number(out,seeds[i][k].y);out<<']';
   }
   out<<']';
  }
  out<<"],\"featureBackend\":";text(out,backend);
  out<<",\"timingsMs\":{\"featureExtraction\":";number(out,timings.extraction);
  out<<",\"matchingHomography\":";number(out,timings.matchingHomography);
  out<<",\"total\":";number(out,clockMs()-start);out<<"}}";output=out.str();return 0;
 }catch(const std::exception&e){error(e);return -1;}
}

// Prepare all proposals on the coordinator in the original trial order. The
// cache stores successful SIFT results only; binary fallback is intentionally
// NOT cached. Once SIFT fails, its sticky failure persists across detections.
EMSCRIPTEN_KEEPALIVE int grass_pool_prepare(const unsigned char*rgba,int w,int h,const double*ref){try{
 pool=PoolState{};pool.started=clockMs();double start=pool.started;pool.c=context(rgba,w,h,ref);pool.total.green=clockMs()-start;
 auto config=configuration(true);std::map<double,std::vector<Quad>>cache;
 for(auto&cfg:config){Timings t;double ts=clockMs();FeatureTimings ft;std::vector<Quad>raw;pool.c.opts=cfg.second;
  grass_progress(pool.raw.size(),config.size(),cfg.first.c_str(),"features");
  if(!cache.count(pool.c.opts.siftRatio)){try{if(siftFailed)throw std::runtime_error("SIFT previously failed");std::vector<float>gf(pool.c.g.begin(),pool.c.g.end());cache[pool.c.opts.siftRatio]=siftSeeds(gf,w,h,pool.c.ref,pool.c.opts.siftRatio,&ft);}catch(const std::exception&){siftFailed=true;try{std::vector<float>gf(pool.c.g.begin(),pool.c.g.end());raw=binarySeeds(gf,w,h,pool.c.ref,&ft);}catch(const std::exception&){raw={};}}}
  if(cache.count(pool.c.opts.siftRatio))raw=cache[pool.c.opts.siftRatio];t.features=clockMs()-ts;t.featureExtraction=ft.extraction;t.matchingHomography=ft.matchingHomography;
  pool.raw.push_back(std::move(raw));pool.timings.push_back(t);
 }
 pool.candidates.resize(config.size());pool.received.assign(config.size(),false);pool.active=true;
 std::ostringstream out;out<<"{\"plans\":[";for(size_t i=0;i<pool.raw.size();i++){if(i)out<<',';out<<"{\"index\":"<<i<<",\"name\":";text(out,config[i].first);out<<",\"raw\":[";for(size_t j=0;j<pool.raw[i].size();j++){if(j)out<<',';out<<'[';for(int k=0;k<4;k++){if(k)out<<',';number(out,pool.raw[i][j][k].x);out<<',';number(out,pool.raw[i][j][k].y);}out<<']';}out<<"]}";}out<<"]}";output=out.str();return 0;
 }catch(const std::exception&e){pool.active=false;error(e);return -1;}}
EMSCRIPTEN_KEEPALIVE int grass_pool_trial(const unsigned char*rgba,int w,int h,const double*ref,int index,const double*raw,int n){try{
 auto config=configuration(true);if(index<0||index>=int(config.size())||n<0||n>4096||(n&&!raw))throw std::runtime_error("Invalid automatic trial");Context c=context(rgba,w,h,ref);c.opts=config[index].second;std::vector<Quad>qs;for(int i=0;i<n;i++){Quad q;for(int k=0;k<4;k++){double x=raw[i*8+k*2],y=raw[i*8+k*2+1];if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("Nonfinite automatic proposal");q[k]={x,y};}qs.push_back(q);}
 Timings t;double start=clockMs();auto r=runDetection(c,qs,false,t);t.total=clockMs()-start;snapshot(r,t);output="{}";return 0;
 }catch(const std::exception&e){error(e);return -1;}}
EMSCRIPTEN_KEEPALIVE const unsigned char*grass_pool_blob(){return blob.data();}
EMSCRIPTEN_KEEPALIVE int grass_pool_blob_size(){return blob.size();}
EMSCRIPTEN_KEEPALIVE int grass_pool_submit(int index,const unsigned char*p,int size){try{requirePool();if(index<0||index>=int(pool.raw.size())||pool.received[index])throw std::runtime_error("Invalid/duplicate automatic trial index");Result r;Timings t;readSnapshot(p,size,r,t);pool.candidates[index]=std::move(r);auto&ft=pool.timings[index];t.features=ft.features;t.featureExtraction=ft.featureExtraction;t.matchingHomography=ft.matchingHomography;t.total+=ft.features;ft=t;pool.received[index]=true;output="{}";return 0;}catch(const std::exception&e){error(e);return -1;}}
// Failure fallback consumes the PREPARED proposals, not another feature pass.
EMSCRIPTEN_KEEPALIVE int grass_pool_finish(){try{requirePool();auto config=configuration(true);Result result;std::vector<Block>merged;std::vector<Trial>trials;Timings total=pool.total;
 for(size_t i=0;i<config.size();i++){auto&t=pool.timings[i];if(!pool.received[i]){pool.c.opts=config[i].second;double start=clockMs();pool.candidates[i]=runDetection(pool.c,pool.raw[i],false,t);t.total=clockMs()-start+t.features;pool.received[i]=true;}auto&candidate=pool.candidates[i];if(i==0){result=candidate;merged=candidate.blocks;}else{double start=clockMs();merged=mergeAligned(merged,candidate.blocks);t.trialUnion+=clockMs()-start;}
 trials.push_back({config[i].first,int(candidate.blocks.size()),int(merged.size()),t});total.features+=t.features;total.featureExtraction+=t.featureExtraction;total.matchingHomography+=t.matchingHomography;total.ecc+=t.ecc;total.latticeFallback+=t.latticeFallback;total.trialUnion+=t.trialUnion;total.verify+=t.verify;total.plane+=t.plane;total.voxel+=t.voxel;
 }result.blocks=merged;result.hasCamera=result.hasPlane=false;total.total=clockMs()-pool.started;output=serialize(result,trials,total,true);pool.active=false;pool.raw.clear();pool.candidates.clear();pool.c.g.clear();return 0;
 }catch(const std::exception&e){pool.active=false;error(e);return -1;}}
EMSCRIPTEN_KEEPALIVE void grass_pool_abort(){pool=PoolState{};blob.clear();}
EMSCRIPTEN_KEEPALIVE int grass_detect(const unsigned char*rgba,int w,int h,const double*ref,const double*opts,const double*seeds,int n,int autoFlag){try{double allStart=clockMs(),start=allStart;Context c=context(rgba,w,h,ref);Timings total;total.green=clockMs()-start;Options base=options(opts);if(n<0||n>4096||(n&&!seeds))throw std::runtime_error("Invalid seed count");std::vector<Quad>manual;for(int i=0;i<n;i++){Quad q;for(int k=0;k<4;k++){double x=seeds[i*8+k*2],y=seeds[i*8+k*2+1];if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("Invalid seed quad");q[k]={x,y};}manual.push_back(canonical(q));}
 auto config=configuration(autoFlag&&n==0,base);
 std::map<double,std::vector<Quad>>cache;Result result;std::vector<Block>merged;std::vector<Trial>trials;for(auto&cfg:config){grass_progress(trials.size(),config.size(),cfg.first.c_str(),"features");double ts=clockMs();Timings t;c.opts=cfg.second;std::vector<Quad>raw=manual;if(!n){start=clockMs();FeatureTimings ft;if(!cache.count(c.opts.siftRatio)){try{if(siftFailed)throw std::runtime_error("SIFT previously failed");std::vector<float>gf(c.g.begin(),c.g.end());cache[c.opts.siftRatio]=siftSeeds(gf,c.w,c.h,c.ref,c.opts.siftRatio,&ft);}catch(const std::exception&){siftFailed=true;try{std::vector<float>gf(c.g.begin(),c.g.end());raw=binarySeeds(gf,c.w,c.h,c.ref,&ft);}catch(const std::exception&){raw={};}}}if(cache.count(c.opts.siftRatio))raw=cache[c.opts.siftRatio];t.features=clockMs()-start;t.featureExtraction=ft.extraction;t.matchingHomography=ft.matchingHomography;}grass_progress(trials.size(),config.size(),cfg.first.c_str(),"refining and growing");Result candidate=runDetection(c,raw,n>0,t);if(trials.empty()){result=candidate;merged=candidate.blocks;}else {double us=clockMs();merged=mergeAligned(merged,candidate.blocks);t.trialUnion+=clockMs()-us;}grass_progress(trials.size()+1,config.size(),cfg.first.c_str(),"complete");t.total=clockMs()-ts;trials.push_back({cfg.first,int(candidate.blocks.size()),int(merged.size()),t});total.features+=t.features;total.ecc+=t.ecc;total.latticeFallback+=t.latticeFallback;total.trialUnion+=t.trialUnion;total.featureExtraction+=t.featureExtraction;total.matchingHomography+=t.matchingHomography;total.verify+=t.verify;total.plane+=t.plane;total.voxel+=t.voxel;}
 result.blocks=merged;if(config.size()>1){result.hasCamera=result.hasPlane=false;}total.total=clockMs()-allStart;output=serialize(result,trials,total,config.size()>1);return 0;
 }catch(const std::exception&e){error(e);return -1;}}
}
