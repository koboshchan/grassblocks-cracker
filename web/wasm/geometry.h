#pragma once
#include "features.h"
#include <vector>
#include <array>
#include <string>
#include <functional>
namespace grass {
using H3=std::array<double,9>;
using Camera=std::array<double,12>;
struct Verdict { bool valid=false; int rotation=0,votes=0; double score=0,margin=0; };
struct Block:Verdict { Quad quad{}; int x=0,y=0,z=0; };
struct Options {double accept=.88,margin=.35,relaxed=.78,coarse=.55,refineMin=-1,refineMargin=-1,refineShift=3.5,siftRatio=.68;double refineVotes=3;bool refitPlane=false,refitFlat=false;};
struct Context {int w,h;std::vector<float>g;std::array<double,256>ref;std::array<std::array<double,256>,4>refs;Options opts;};
struct Plane {bool valid=false;H3 H{};std::vector<int>ids;};
struct Result {std::vector<Block>blocks,seeds;std::string reason,mode;Camera camera{};H3 planeH{};bool hasCamera=false,hasPlane=false;std::vector<std::string>notes;};
struct Timings {double green=0,features=0,featureExtraction=0,matchingHomography=0,ecc=0,verify=0,plane=0,voxel=0,latticeFallback=0,trialUnion=0,total=0;};
using Refiner=std::function<bool(const Quad&,int,Quad&)>;
bool homography(const std::vector<Point>&src,const std::vector<Point>&dst,H3&out);
Point applyH(const H3&H,double x,double y);
bool inv3(const H3&m,H3&o);
Verdict classify(const Context&ctx,const Quad&q);
std::vector<Verdict> classifyBatchNative(const Context&ctx,const std::vector<Quad>&qs);
Result runDetection(Context&ctx,const std::vector<Quad>&raw,bool manual,Timings&t);
std::vector<Block>mergeAligned(const std::vector<Block>&original,const std::vector<Block>&incoming);
double clockMs();
}
