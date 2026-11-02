#pragma once
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <csignal>
#include <algorithm>
struct Block { int32_t x,y,z,r,side; };
static volatile std::sig_atomic_t stopped=0;
static void stop_search(int){stopped=1;}
static int64_t number(const std::string&s){size_t n=0;long long v=std::stoll(s,&n);if(n!=s.size())throw std::runtime_error("Invalid integer: "+s);return v;}
static int32_t coord(int64_t v){if(v<INT32_MIN||v>INT32_MAX)throw std::runtime_error("Coordinate outside int32");return int32_t(v);}
struct Options{
 int32_t lo[3]={-6000,60,-6000}, hi[3]={6000,256,6000};
 uint64_t dim[3],total=1; std::string path="formation.txt";std::vector<int> north{0};
};
static Options options(int argc,char**argv){
 Options o;std::vector<std::string> pos;
 for(int i=1;i<argc;i++){std::string a=argv[i];
 if(a=="--help"||a=="-h"){std::cout<<"Usage: solver [--formation FILE] [--north 0|1|2|3|all] [xmin xmax ymin ymax zmin zmax]\nInclusive int32 bounds; stdout: x y z north. Progress, rate and total on stderr.\n";std::exit(0);}
 else if(a=="--formation"||a=="--north"){if(++i==argc)throw std::runtime_error("Missing option value");if(a=="--formation")o.path=argv[i];else {std::string n=argv[i];if(n=="all")o.north={0,1,2,3};else{auto v=number(n);if(v<0||v>3)throw std::runtime_error("north must be 0..3 or all");o.north={int(v)};}}}
 else pos.push_back(a);
 }
 if(!pos.empty()&&pos.size()!=6)throw std::runtime_error("Expected six positional bounds");
 for(int i=0;i<3;i++){if(!pos.empty()){o.lo[i]=coord(number(pos[2*i]));o.hi[i]=coord(number(pos[2*i+1]));}if(o.lo[i]>o.hi[i])throw std::runtime_error("Reversed bounds");o.dim[i]=uint64_t(int64_t(o.hi[i])-o.lo[i])+1;if(o.total>UINT64_MAX/o.dim[i])throw std::runtime_error("Search volume overflows uint64");o.total*=o.dim[i];}
 if(o.total>UINT64_MAX/o.north.size())throw std::runtime_error("Combined north volume overflows uint64");
 return o;
}
static std::vector<Block> load(const std::string&path){
 std::ifstream in(path);if(!in)throw std::runtime_error("Cannot open "+path);
 std::vector<Block>b;std::string line;int64_t count=-1;size_t lineno=0;
 while(std::getline(in,line)){lineno++;line=line.substr(0,line.find('#'));std::istringstream s(line);std::string first;if(!(s>>first))continue;
 if(count<0){count=number(first);std::string extra;if(s>>extra)throw std::runtime_error("Count line must contain one integer");if(count<=0||count>1000000)throw std::runtime_error("Formation count must be 1..1000000");continue;}
 if(b.size()>=size_t(count))throw std::runtime_error("Extra formation rows");
 std::string sy,sz,sr,ss;if(!(s>>sy>>sz>>sr>>ss))throw std::runtime_error("Incomplete row "+std::to_string(lineno));
 Block v{coord(number(first)),coord(number(sy)),coord(number(sz)),coord(number(sr)),coord(number(ss))};
 if(v.r<0||v.r>3||v.side<0||v.side>1||(v.side&&v.r>1))throw std::runtime_error("Invalid rotation/is_side at row "+std::to_string(lineno));
 b.push_back(v); // Remaining confidence metadata intentionally ignored.
 }if(count<0||b.size()!=size_t(count))throw std::runtime_error("Missing rows or empty formation");return b;
}
static std::vector<Block> orient(std::vector<Block>b,int n,const Options&o){
 for(auto &v:b){int64_t x=v.x,z=v.z;for(int j=0;j<n;j++){int64_t old=x;x=-z;z=old;}v.x=coord(x);v.z=coord(z);v.r=(v.r+n)&3;if(v.side)v.r&=1;
 for(int i=0;i<3;i++){int32_t d=i==0?v.x:i==1?v.y:v.z;coord(int64_t(o.lo[i])+d);coord(int64_t(o.hi[i])+d);}
 }return b;
}
static void decode(uint64_t id,const Options&o,int32_t &x,int32_t &y,int32_t &z){
 y=int32_t(int64_t(o.lo[1])+int64_t(id%o.dim[1]));id/=o.dim[1];
 z=int32_t(int64_t(o.lo[2])+int64_t(id%o.dim[2]));id/=o.dim[2];
 x=int32_t(int64_t(o.lo[0])+int64_t(id));
}
