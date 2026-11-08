#include "geometry.h"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace grass;
int main(){
 H3 H;std::vector<Point>s={{0,0},{1,0},{1,1},{0,1}},d={{11,13},{43,15},{48,52},{9,49}};assert(homography(s,d,H));for(int i=0;i<4;i++){auto p=applyH(H,s[i].x,s[i].y);assert(std::hypot(p.x-d[i].x,p.y-d[i].y)<1e-9);}H3 I;assert(inv3(H,I));for(int i=0;i<4;i++){auto p=applyH(I,d[i].x,d[i].y);assert(std::hypot(p.x-s[i].x,p.y-s[i].y)<1e-9);}H3 singular{};assert(!inv3(singular,I));assert(!homography({{0,0},{0,0},{0,0},{0,0}},d,H));
 std::vector<Block>a,b;for(int i=0;i<3;i++){Block v;v.x=i;v.y=-2;v.z=7;v.rotation=i;v.quad={Point(i*30,0),Point(i*30+24,0),Point(i*30+24,24),Point(i*30,24)};a.push_back(v);v.x-=4;v.y+=3;v.z-=9;b.push_back(v);}Block extra=b.back();extra.x+=1;for(auto&p:extra.quad)p.x+=30;b.push_back(extra);auto merged=mergeAligned(a,b);assert(merged.size()==4);assert(merged.back().x==3&&merged.back().y==-2&&merged.back().z==7);b[0].rotation=3;assert(mergeAligned(a,b).size()==3);
 Context c;c.w=c.h=40;c.g.resize(1600);for(int y=0;y<40;y++)for(int x=0;x<40;x++)c.g[y*40+x]=(x*13+y*29+x*y*7)%256;for(int y=0;y<16;y++)for(int x=0;x<16;x++){// sampling at integer points with quad offset -.5
 c.ref[y*16+x]=c.g[(y+2)*40+x+2];}for(int k=0;k<4;k++)for(int y=0;y<16;y++)for(int x=0;x<16;x++){int xx=x,yy=y;for(int j=0;j<k;j++){int t=xx;xx=yy;yy=15-t;}c.refs[k][y*16+x]=c.ref[yy*16+xx];}Quad q={Point(1.5,1.5),Point(17.5,1.5),Point(17.5,17.5),Point(1.5,17.5)};auto v=classify(c,q);std::cerr<<v.valid<<" "<<v.rotation<<" "<<v.votes<<" "<<v.score<<"\n";assert(v.valid&&v.rotation==0&&v.votes==4&&v.score>.999999);q[0]={-100,-100};assert(!classify(c,q).valid);
 std::cout<<"native geometry homography/inverse/singularity, union translation/conflict, double classifier passed\n";
}
