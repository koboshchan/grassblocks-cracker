#include "geometry.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <chrono>
#include <memory>
#include <set>
#include <map>
#include <unordered_map>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif
namespace grass {
double clockMs(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static const Quad unit={Point(0,0),Point(1,0),Point(1,1),Point(0,1)};
static bool solve(std::vector<std::vector<double>> A,std::vector<double>b,std::vector<double>&out){int n=b.size();for(int i=0;i<n;i++)A[i].push_back(b[i]);for(int j=0;j<n;j++){int k=j;for(int i=j+1;i<n;i++)if(std::abs(A[i][j])>std::abs(A[k][j]))k=i;std::swap(A[j],A[k]);if(std::abs(A[j][j])<1e-12)return false;double d=A[j][j];for(int t=j;t<=n;t++)A[j][t]/=d;for(int i=0;i<n;i++)if(i!=j){double f=A[i][j];if(f)for(int t=j;t<=n;t++)A[i][t]-=f*A[j][t];}}out.resize(n);for(int i=0;i<n;i++)out[i]=A[i][n];return true;}
bool homography(const std::vector<Point>&s,const std::vector<Point>&d,H3&H){std::vector<std::vector<double>>N(8,std::vector<double>(8));std::vector<double>r(8),o;for(size_t i=0;i<s.size();i++){double x=s[i].x,y=s[i].y,u=d[i].x,v=d[i].y;double rows[2][9]={{x,y,1,0,0,0,-u*x,-u*y,u},{0,0,0,x,y,1,-v*x,-v*y,v}};for(auto&q:rows)for(int a=0;a<8;a++){r[a]+=q[a]*q[8];for(int b=0;b<8;b++)N[a][b]+=q[a]*q[b];}}if(!solve(N,r,o))return false;std::copy(o.begin(),o.end(),H.begin());H[8]=1;return true;}
Point applyH(const H3&H,double x,double y){double d=H[6]*x+H[7]*y+H[8];return {(H[0]*x+H[1]*y+H[2])/d,(H[3]*x+H[4]*y+H[5])/d};}
bool inv3(const H3&m,H3&o){double a=m[0],b=m[1],c=m[2],d=m[3],e=m[4],f=m[5],g=m[6],h=m[7],i=m[8],A=e*i-f*h,B=-(d*i-f*g),C=d*h-e*g,det=a*A+b*B+c*C;if(std::abs(det)<1e-18)return false;o={A/det,-(b*i-c*h)/det,(b*f-c*e)/det,B/det,(a*i-c*g)/det,-(a*f-c*d)/det,C/det,-(a*h-b*g)/det,(a*e-b*d)/det};return true;}
static H3 mul3(const H3&a,const H3&b){H3 o{};for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)o[i*3+j]+=a[i*3+k]*b[k*3+j];return o;}
static double dist(Point a,Point b){return std::hypot(a.x-b.x,a.y-b.y);}
static double minSide(const Quad&q){double m=INFINITY;for(int i=0;i<4;i++)m=std::min(m,dist(q[i],q[(i+1)%4]));return m;}
static bool convex(const Quad&q){int s=0;for(int i=0;i<4;i++){Point a=q[i],b=q[(i+1)%4],c=q[(i+2)%4];double z=(b.x-a.x)*(c.y-b.y)-(b.y-a.y)*(c.x-b.x);if(!z)return false;int v=z>0?1:-1;if(!s)s=v;else if(s!=v)return false;}return true;}
static bool inImage(const Quad&q,int w,int h){for(auto p:q)if(!std::isfinite(p.x)||!std::isfinite(p.y)||p.x<0||p.y<0||p.x>=w-1||p.y>=h-1)return false;return true;}
static Quad gridQuad(const H3&H,int x,int z){Quad q;for(int i=0;i<4;i++)q[i]=applyH(H,x+unit[i].x,z+unit[i].y);return q;}
static double corrIds(const float*patch,const double*ref,const int*ids,int count){
 double sa=0,sb=0,aa=0,bb=0,ab=0;
 for(int t=0;t<count;t++){int idx=ids[t];double a=patch[idx],b=ref[idx];sa+=a;sb+=b;aa+=a*a;bb+=b*b;ab+=a*b;}
 double n=count,va=aa-sa*sa/n,vb=bb-sb*sb/n;
 if(va<1e-9||vb<1e-9)return 0;
 return (ab-sa*sb/n)/std::sqrt(va*vb);
}
namespace {
struct ClassifyTables {
 int inner[196];
 int quads[4][49];
 ClassifyTables(){
  int nInner=0,nQuad[4]={0,0,0,0};
  for(int j=1;j<15;j++)for(int i=1;i<15;i++){
   int idx=j*16+i;
   inner[nInner++]=idx;
   int qid=(i>=8?1:0)+(j>=8?2:0);
   quads[qid][nQuad[qid]++]=idx;
  }
 }
};
const ClassifyTables&classifyTables(){static const ClassifyTables tbl;return tbl;}
}
Verdict classify(const Context&c,const Quad&q){
 H3 H;std::vector<Point>s={{0,0},{16,0},{16,16},{0,16}},d(q.begin(),q.end());
 if(!homography(s,d,H))return {};
 float patch[256];const uint8_t*g=c.g.data();
 for(int j=0;j<16;j++)for(int i=0;i<16;i++){
  Point p=applyH(H,i+.5,j+.5);
  if(!(p.x>=0&&p.y>=0&&p.x<c.w-1&&p.y<c.h-1))return {};
  int xx=std::floor(p.x),yy=std::floor(p.y);
  double fx=p.x-xx,fy=p.y-yy;
  patch[j*16+i]=g[yy*c.w+xx]*(1-fx)*(1-fy)+g[yy*c.w+xx+1]*fx*(1-fy)+g[(yy+1)*c.w+xx]*(1-fx)*fy+g[(yy+1)*c.w+xx+1]*fx*fy;
 }
 const auto&tbl=classifyTables();
 std::array<double,4>sc;
 for(int k=0;k<4;k++)sc[k]=corrIds(patch,c.refs[k].data(),tbl.inner,196);
 std::array<int,4>order={0,1,2,3};
 std::stable_sort(order.begin(),order.end(),[&](int a,int b){return sc[a]>sc[b];});
 int k=order[0],votes=0;
 for(int qid=0;qid<4;qid++){
  int best=0;double bestScore=-1e30;
  for(int a=0;a<4;a++){
   double score=corrIds(patch,c.refs[a].data(),tbl.quads[qid],49);
   if(a==0||score>bestScore){bestScore=score;best=a;}
  }
  if(best==k)votes++;
 }
 return {true,k,votes,sc[k],sc[k]-sc[order[1]]};
}
// Batch verification deliberately uses the existing Float32 ABI and closed-form
// kernel; single/ECC verification uses the double precision reference above.
extern "C" {void set_refs(const unsigned char*);void classify_batch(const unsigned char*,int,int,const float*,int,float*);}
std::vector<Verdict> classifyBatchNative(const Context&c,const std::vector<Quad>&qs){static thread_local bool hasRefs=false;static thread_local std::array<unsigned char,1024>cachedRefs{};bool match=hasRefs;for(int k=0;k<4&&match;k++)for(int i=0;i<256;i++)if(static_cast<unsigned char>(c.refs[k][i])!=cachedRefs[k*256+i]){match=false;break;}if(!match){for(int k=0;k<4;k++)for(int i=0;i<256;i++)cachedRefs[k*256+i]=c.refs[k][i];set_refs(cachedRefs.data());hasRefs=true;}std::vector<float>flat(qs.size()*8),out(qs.size()*5);for(size_t j=0;j<qs.size();j++)for(int i=0;i<4;i++){flat[j*8+i*2]=qs[j][i].x;flat[j*8+i*2+1]=qs[j][i].y;}classify_batch(c.g.data(),c.w,c.h,flat.data(),qs.size(),out.data());std::vector<Verdict>r(qs.size());for(size_t i=0;i<qs.size();i++)r[i]={out[i*5]!=0,int(out[i*5+1]),int(out[i*5+4]),out[i*5+2],out[i*5+3]};return r;}
namespace {
struct EccKey {
 std::array<int64_t, 8> q;
 int rot;
 bool operator==(const EccKey& o) const { return rot == o.rot && q == o.q; }
};

struct EccKeyHash {
 size_t operator()(const EccKey& k) const noexcept {
  uint64_t h = uint64_t(k.rot) * 0x9e3779b97f4a7c15ULL;
  for (int64_t v : k.q) h ^= uint64_t(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return size_t(h);
 }
};

struct EccCached {
 bool ok = false;
 Quad nq{};
};

inline float dotF32(const float* a, const float* b, int n) {
#ifdef __wasm_simd128__
 v128_t acc0 = wasm_f32x4_splat(0.0f);
 v128_t acc1 = wasm_f32x4_splat(0.0f);
 int i = 0;
 for (; i + 8 <= n; i += 8) {
  acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
  acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
 }
 v128_t sum = wasm_f32x4_add(acc0, acc1);
 sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 2, 3, 0, 1));
 sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2));
 float s = wasm_f32x4_extract_lane(sum, 0);
 for (; i < n; ++i) s += a[i] * b[i];
 return s;
#else
 float s = 0.0f;
 for (int i = 0; i < n; ++i) s += a[i] * b[i];
 return s;
#endif
}

bool invert8x8(const double A[8][8], double inv[8][8]) {
 double aug[8][16]{};
 for (int i = 0; i < 8; ++i) {
  for (int j = 0; j < 8; ++j) aug[i][j] = A[i][j];
  aug[i][8 + i] = 1.0;
 }
 for (int col = 0; col < 8; ++col) {
  int piv = col;
  for (int r = col + 1; r < 8; ++r) {
   if (std::abs(aug[r][col]) > std::abs(aug[piv][col])) piv = r;
  }
  if (std::abs(aug[piv][col]) < 1e-12) return false;
  if (piv != col) {
   for (int j = 0; j < 16; ++j) std::swap(aug[col][j], aug[piv][j]);
  }
  double d = aug[col][col];
  for (int j = col; j < 16; ++j) aug[col][j] /= d;
  for (int r = 0; r < 8; ++r) {
   if (r == col) continue;
   double f = aug[r][col];
   if (f == 0.0) continue;
   for (int j = col; j < 16; ++j) aug[r][j] -= f * aug[col][j];
  }
 }
 for (int i = 0; i < 8; ++i) {
  for (int j = 0; j < 8; ++j) inv[i][j] = aug[i][8 + j];
 }
 return true;
}
} // namespace

// Precomputed Inverse-Compositional ECC on 64x64 upscaled grass templates.
// Eliminates per-iteration blurring, gradient, Jacobian, and Hessian solves.
class ECC {
 static constexpr int N = 64;
 static constexpr int M = (N - 2) * (N - 2); // 62 * 62 = 3844 (multiple of 4)
 int w = 0, h = 0;
 std::vector<float> img;
 struct Template {
  bool valid = false;
  std::vector<float> tBar;
  std::array<std::vector<float>, 8> pMat;
 };
 std::array<Template, 4> templates;
 std::vector<float> patchBuf;
 std::unordered_map<EccKey, EccCached, EccKeyHash> cache;

 static void blurSeparable121(std::vector<double>& grid) {
  std::vector<double> tmp(N * N, 0.0);
  for (int y = 0; y < N; ++y) {
   for (int x = 0; x < N; ++x) {
    int xm = std::max(0, x - 1), xp = std::min(N - 1, x + 1);
    tmp[y * N + x] = 0.25 * grid[y * N + xm] + 0.5 * grid[y * N + x] + 0.25 * grid[y * N + xp];
   }
  }
  for (int y = 0; y < N; ++y) {
   int ym = std::max(0, y - 1), yp = std::min(N - 1, y + 1);
   for (int x = 0; x < N; ++x) {
    grid[y * N + x] = 0.25 * tmp[ym * N + x] + 0.5 * tmp[y * N + x] + 0.25 * tmp[yp * N + x];
   }
  }
 }

 void buildTemplate(int rot, const std::array<double, 256>& ref) {
  std::vector<double> grid(N * N, 0.0);
  for (int y = 0; y < N; ++y) {
   for (int x = 0; x < N; ++x) {
    grid[y * N + x] = ref[(y >> 2) * 16 + (x >> 2)] / 255.0;
   }
  }
  // Two passes of [1,2,1]/4 produce a smooth [1,4,6,4,1]/16 kernel (~0.35 texel sigma).
  blurSeparable121(grid);
  blurSeparable121(grid);

  // Zero-mean unit-norm template vector over interior [1..N-2]^2.
  double meanT = 0.0;
  std::vector<double> tVec(M, 0.0);
  int idx = 0;
  for (int y = 1; y < N - 1; ++y) {
   for (int x = 1; x < N - 1; ++x) {
    double v = grid[y * N + x];
    tVec[idx++] = v;
    meanT += v;
   }
  }
  meanT /= M;
  double norm2 = 0.0;
  for (int i = 0; i < M; ++i) {
   tVec[i] -= meanT;
   norm2 += tVec[i] * tVec[i];
  }
  if (norm2 < 1e-12) return;
  double invNorm = 1.0 / std::sqrt(norm2);
  templates[rot].tBar.resize(M);
  for (int i = 0; i < M; ++i) {
   tVec[i] *= invNorm;
   templates[rot].tBar[i] = static_cast<float>(tVec[i]);
  }

  // Steepest-descent images J[0..7] in normalized [-1, 1]^2 coordinates,
  // zero-meaned and projected orthogonal to tBar.
  const double invTwoStep = double(N) / 4.0; // 1 / (2 * (2 / N))
  std::vector<double> J(8 * M, 0.0);
  idx = 0;
  for (int y = 1; y < N - 1; ++y) {
   double eta = (2.0 * (y + 0.5)) / N - 1.0;
   for (int x = 1; x < N - 1; ++x) {
    double xi = (2.0 * (x + 0.5)) / N - 1.0;
    double gx = (grid[y * N + (x + 1)] - grid[y * N + (x - 1)]) * invTwoStep * invNorm;
    double gy = (grid[(y + 1) * N + x] - grid[(y - 1) * N + x]) * invTwoStep * invNorm;
    J[0 * M + idx] = gx * xi;
    J[1 * M + idx] = gx * eta;
    J[2 * M + idx] = gx;
    J[3 * M + idx] = gy * xi;
    J[4 * M + idx] = gy * eta;
    J[5 * M + idx] = gy;
    J[6 * M + idx] = -xi * (gx * xi + gy * eta);
    J[7 * M + idx] = -eta * (gx * xi + gy * eta);
    ++idx;
   }
  }
  for (int k = 0; k < 8; ++k) {
   double* jk = J.data() + k * M;
   double meanK = 0.0;
   for (int i = 0; i < M; ++i) meanK += jk[i];
   meanK /= M;
   double proj = 0.0;
   for (int i = 0; i < M; ++i) {
    jk[i] -= meanK;
    proj += jk[i] * tVec[i];
   }
   for (int i = 0; i < M; ++i) jk[i] -= proj * tVec[i];
  }

  double H[8][8]{}, Hinv[8][8]{};
  for (int a = 0; a < 8; ++a) {
   const double* ja = J.data() + a * M;
   for (int b = a; b < 8; ++b) {
    const double* jb = J.data() + b * M;
    double s = 0.0;
    for (int i = 0; i < M; ++i) s += ja[i] * jb[i];
    H[a][b] = H[b][a] = s;
   }
   H[a][a] += 1e-7; // tiny Tikhonov regularization for numerical stability
  }
  if (!invert8x8(H, Hinv)) return;

  for (int a = 0; a < 8; ++a) {
   templates[rot].pMat[a].resize(M);
   for (int i = 0; i < M; ++i) {
    double s = 0.0;
    for (int b = 0; b < 8; ++b) s += Hinv[a][b] * J[b * M + i];
    templates[rot].pMat[a][i] = static_cast<float>(s);
   }
  }
  templates[rot].valid = true;
 }

 bool samplePatch(const H3& W, std::vector<float>& out, double& sumW, double& sumWW) const {
  const float* src = img.data();
  const int maxX = w - 1, maxY = h - 1;
  const double step = 2.0 / N;
  const double xiStart = (2.0 * 1.5) / N - 1.0;
  double sw = 0.0, sww = 0.0;
  int idx = 0;
  for (int y = 1; y < N - 1; ++y) {
   double eta = (2.0 * (y + 0.5)) / N - 1.0;
   double rx = W[0] * xiStart + W[1] * eta + W[2];
   double ry = W[3] * xiStart + W[4] * eta + W[5];
   double rd = W[6] * xiStart + W[7] * eta + W[8];
   double dx = W[0] * step, dy = W[3] * step, dd = W[6] * step;
   for (int x = 1; x < N - 1; ++x) {
    if (std::abs(rd) < 1e-12) return false;
    double invD = 1.0 / rd;
    double px = rx * invD;
    double py = ry * invD;
    if (px < 0.0) px = 0.0;
    else if (px > maxX) px = maxX;
    if (py < 0.0) py = 0.0;
    else if (py > maxY) py = maxY;
    int ix = int(px), iy = int(py);
    int ix1 = std::min(ix + 1, maxX), iy1 = std::min(iy + 1, maxY);
    float fx = float(px - ix), fy = float(py - iy);
    const float* r0 = src + iy * w;
    const float* r1 = src + iy1 * w;
    float v0 = r0[ix] + (r0[ix1] - r0[ix]) * fx;
    float v1 = r1[ix] + (r1[ix1] - r1[ix]) * fx;
    float v = v0 + (v1 - v0) * fy;
    out[idx++] = v;
    sw += v;
    sww += double(v) * double(v);
    rx += dx;
    ry += dy;
    rd += dd;
   }
  }
  sumW = sw;
  sumWW = sww;
  return true;
 }

public:
 ECC(const Context& c) : w(c.w), h(c.h), img(c.g.size()), patchBuf(M, 0.0f) {
  for (size_t i = 0; i < c.g.size(); ++i) img[i] = c.g[i] * (1.0f / 255.0f);
  for (int k = 0; k < 4; ++k) buildTemplate(k, c.refs[k]);
 }

 bool operator()(const Quad& q, int rot, Quad& nq) {
  if (rot < 0 || rot >= 4 || !templates[rot].valid) return false;
  EccKey key{{}, rot};
  for (int i = 0; i < 4; ++i) {
   key.q[i * 2] = std::llround(q[i].x * 8.0);
   key.q[i * 2 + 1] = std::llround(q[i].y * 8.0);
  }
  auto it = cache.find(key);
  if (it != cache.end()) {
   nq = it->second.nq;
   return it->second.ok;
  }

  static const std::vector<Point> normCorners = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
  std::vector<Point> d(q.begin(), q.end());
  H3 W;
  if (!homography(normCorners, d, W)) {
   if (cache.size() < 32768) cache.emplace(key, EccCached{false, {}});
   return false;
  }

  const auto& tpl = templates[rot];
  auto& patch = patchBuf;
  H3 bestW = W;
  double bestRho = -2.0;
  int drops = 0;

  for (int iter = 0; iter < 30; ++iter) {
   double sw = 0.0, sww = 0.0;
   if (!samplePatch(W, patch, sw, sww)) break;
   double var = sww - (sw * sw) / M;
   if (var < 1e-10) break;
   double tw = dotF32(tpl.tBar.data(), patch.data(), M);
   double rho = tw / std::sqrt(var);
   if (rho > bestRho) {
    bestRho = rho;
    bestW = W;
    drops = 0;
   } else {
    if (++drops >= 2) break;
   }
   if (tw <= 1e-6) break;

   double invTw = 1.0 / tw;
   double dp[8];
   double maxStep = 0.0;
   for (int k = 0; k < 8; ++k) {
    dp[k] = dotF32(tpl.pMat[k].data(), patch.data(), M) * invTw;
    maxStep = std::max(maxStep, std::abs(dp[k]));
   }
   if (!std::isfinite(maxStep)) break;
   if (maxStep > 0.25) {
    double scale = 0.25 / maxStep;
    for (int k = 0; k < 8; ++k) dp[k] *= scale;
   }
   H3 dW = {
    1.0 + dp[0], dp[1], dp[2],
    dp[3], 1.0 + dp[4], dp[5],
    dp[6], dp[7], 1.0
   };
   H3 dWi;
   if (!inv3(dW, dWi)) break;
   W = mul3(W, dWi);
   if (std::abs(W[8]) < 1e-12) break;
   double invW8 = 1.0 / W[8];
   for (double& v : W) v *= invW8;
   if (maxStep < 1e-4) {
    // One final evaluation if converged cleanly on this step.
    double sw2 = 0.0, sww2 = 0.0;
    if (samplePatch(W, patch, sw2, sww2)) {
     double var2 = sww2 - (sw2 * sw2) / M;
     if (var2 >= 1e-10) {
      double rho2 = dotF32(tpl.tBar.data(), patch.data(), M) / std::sqrt(var2);
      if (rho2 > bestRho) { bestRho = rho2; bestW = W; }
     }
    }
    break;
   }
  }

  bool ok = bestRho > 0.0;
  if (ok) {
   for (int i = 0; i < 4; ++i) nq[i] = applyH(bestW, normCorners[i].x, normCorners[i].y);
  }
  if (cache.size() < 32768) cache.emplace(key, EccCached{ok, nq});
  return ok;
 }
};

Refiner makeEccRefiner(const Context& c) {
 auto ecc = std::make_shared<ECC>(c);
 return [ecc](const Quad& q, int rot, Quad& nq) { return (*ecc)(q, rot, nq); };
}
static Block verify(const Context&c,Quad q,const Refiner&refine,Verdict r,bool supplied=false){if(!supplied)r=classify(c,q);Block b;if(!r.valid)return b;auto&t=c.opts;double rmin=std::isnan(t.refineMin)?t.relaxed:t.refineMin,rmargin=std::isnan(t.refineMargin)?t.margin:t.refineMargin;if(refine&&r.score>=rmin&&r.margin>=rmargin&&r.votes>=t.refineVotes){Quad nq;if(refine(q,r.rotation,nq)&&convex(nq)&&inImage(nq,c.w,c.h)){double shift=0;for(int i=0;i<4;i++)shift=std::max(shift,dist(nq[i],q[i]));if(shift<t.refineShift){auto r2=classify(c,nq);if(r2.valid&&r2.score>r.score){q=nq;r=r2;}}}}static_cast<Verdict&>(b)=r;b.quad=q;return b;}
static Plane planeFromSeeds(const std::vector<Block>&seeds){Plane best;std::vector<Point>u(unit.begin(),unit.end());for(auto&s:seeds){H3 H0,I;if(!homography(u,std::vector<Point>(s.quad.begin(),s.quad.end()),H0)||!inv3(H0,I))continue;std::vector<int>ids;std::vector<Point>src,dst;for(size_t k=0;k<seeds.size();k++){Quad loc;for(int i=0;i<4;i++)loc[i]=applyH(I,seeds[k].quad[i].x,seeds[k].quad[i].y);double ox=std::floor(loc[0].x+.5),oz=std::floor(loc[0].y+.5),err=0;for(int i=0;i<4;i++)err=std::max(err,dist(loc[i],unit[i]+Point(ox,oz)));if(err<.2){ids.push_back(k);for(int i=0;i<4;i++){src.push_back(unit[i]+Point(ox,oz));dst.push_back(seeds[k].quad[i]);}}}H3 H;if(src.size()>=4&&homography(src,dst,H)&&(!best.valid||ids.size()>best.ids.size()))best={true,H,ids};}return best;}
static std::vector<Block>growPlane(const Context&c,const H3&H,const Refiner&r,int range){std::vector<Block>cand,out;std::vector<Quad>qs;for(int z=-range;z<=range;z++)for(int x=-range;x<=range;x++){Quad q=gridQuad(H,x,z);if(!inImage(q,c.w,c.h)||!convex(q)||minSide(q)<10)continue;Block b;b.quad=q;b.x=x;b.z=z;cand.push_back(b);qs.push_back(q);}auto vs=classifyBatchNative(c,qs);for(size_t i=0;i<cand.size();i++){auto b=verify(c,cand[i].quad,r,vs[i],true);if(b.valid&&b.score>=c.opts.accept&&b.margin>=c.opts.margin&&b.votes==4){b.x=cand[i].x;b.z=cand[i].z;out.push_back(b);}}return out;}
static bool cameraFromPlane(const H3&H,int w,int h,Camera&P,double&f){double cx=w/2.,cy=h/2.,v1x=H[0]/H[6],v1y=H[3]/H[6],v2x=H[1]/H[7],v2y=H[4]/H[7];if(!std::isfinite(v1x+v1y+v2x+v2y))return false;double f2=-((v1x-cx)*(v2x-cx)+(v1y-cy)*(v2y-cy));if(!(f2>100))return false;f=std::sqrt(f2);std::array<double,3>a={(H[0]-cx*H[6])/f,(H[3]-cy*H[6])/f,H[6]},c={(H[1]-cx*H[7])/f,(H[4]-cy*H[7])/f,H[7]},b={c[1]*a[2]-c[2]*a[1],c[2]*a[0]-c[0]*a[2],c[0]*a[1]-c[1]*a[0]};auto norm=[](auto a){return std::hypot(a[0],a[1],a[2]);};double s=(norm(a)+norm(c))/2/norm(b);for(double&v:b)v*=s;P={H[0],f*b[0]+cx*b[2],H[1],H[2],H[3],f*b[1]+cy*b[2],H[4],H[5],H[6],b[2],H[7],H[8]};if((P[7]+P[5])/(P[11]+P[9])>P[7]/P[11])for(int i:{1,5,9})P[i]=-P[i];return true;}
static bool projectQuad(const Camera&P,int x,int y,int z,int w,int h,Quad&q){for(int i=0;i<4;i++){double X=x+unit[i].x,Z=z+unit[i].y,d=P[8]*X+P[9]*y+P[10]*Z+P[11];if(d<=0)return false;double u=(P[0]*X+P[1]*y+P[2]*Z+P[3])/d;if(!(u>=0&&u<w-1))return false;double v=(P[4]*X+P[5]*y+P[6]*Z+P[7])/d;if(!(v>=0&&v<h-1))return false;q[i]={u,v};}if(!convex(q))return false;for(int i=0;i<4;i++){Point d=q[i]-q[(i+1)%4];if(d.x*d.x+d.y*d.y<64)return false;}return true;}
static void jacobi(std::vector<std::vector<double>> A,int n,std::vector<double>&vals,std::vector<std::vector<double>>&V){V.assign(n,std::vector<double>(n));for(int i=0;i<n;i++)V[i][i]=1;for(int sweep=0;sweep<60;sweep++){double off=0;for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)off+=A[i][j]*A[i][j];if(off<1e-24)break;for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){if(std::abs(A[p][q])<1e-30)continue;double th=(A[q][q]-A[p][p])/(2*A[p][q]),t=(th<0?-1.:1.)/(std::abs(th)+std::sqrt(th*th+1)),c=1/std::sqrt(t*t+1),s=t*c;for(int k=0;k<n;k++){double ap=A[k][p],aq=A[k][q];A[k][p]=c*ap-s*aq;A[k][q]=s*ap+c*aq;}for(int k=0;k<n;k++){double ap=A[p][k],aq=A[q][k];A[p][k]=c*ap-s*aq;A[q][k]=s*ap+c*aq;}for(int k=0;k<n;k++){double vp=V[k][p],vq=V[k][q];V[k][p]=c*vp-s*vq;V[k][q]=s*vp+c*vq;}}}vals.resize(n);for(int i=0;i<n;i++)vals[i]=A[i][i];}
static bool dlt(const std::vector<std::array<double,3>>&pts,const std::vector<Point>&pix,Camera&out){double n=pts.size();std::array<double,3>c3{};Point c2{};for(auto p:pts)for(int i=0;i<3;i++)c3[i]+=p[i]/n;for(auto p:pix){c2.x+=p.x/n;c2.y+=p.y/n;}double d3=0,d2=0;for(auto p:pts)d3+=std::hypot(p[0]-c3[0],p[1]-c3[1],p[2]-c3[2])/n;for(auto p:pix)d2+=dist(p,c2)/n;double s3=std::sqrt(3)/(d3?d3:1),s2=std::sqrt(2)/(d2?d2:1);std::vector<std::vector<double>>S(12,std::vector<double>(12));for(size_t i=0;i<pts.size();i++){std::array<double,4>v={(pts[i][0]-c3[0])*s3,(pts[i][1]-c3[1])*s3,(pts[i][2]-c3[2])*s3,1};double u=(pix[i].x-c2.x)*s2,w=(pix[i].y-c2.y)*s2;std::array<double,12>r1{},r2{};for(int k=0;k<4;k++){r1[k]=v[k];r1[k+8]=-u*v[k];r2[k+4]=v[k];r2[k+8]=-w*v[k];}for(auto&r:{r1,r2})for(int a=0;a<12;a++)for(int b=0;b<12;b++)S[a][b]+=r[a]*r[b];}std::vector<double>vals;std::vector<std::vector<double>>vecs;jacobi(S,12,vals,vecs);int mi=0;for(int i=0;i<12;i++)if(vals[i]<vals[mi])mi=i;Camera pn;for(int i=0;i<12;i++)pn[i]=vecs[i][mi];double T3[4][4]={{s3,0,0,-s3*c3[0]},{0,s3,0,-s3*c3[1]},{0,0,s3,-s3*c3[2]},{0,0,0,1}},T2i[3][3]={{1/s2,0,c2.x},{0,1/s2,c2.y},{0,0,1}},tmp[3][4]{};for(int i=0;i<3;i++)for(int j=0;j<4;j++)for(int k=0;k<4;k++)tmp[i][j]+=pn[i*4+k]*T3[k][j];out.fill(0);for(int i=0;i<3;i++)for(int j=0;j<4;j++)for(int k=0;k<3;k++)out[i*4+j]+=T2i[i][k]*tmp[k][j];double sc=out[11];if(std::abs(sc)<1e-12)return false;for(double&v:out)v/=sc;return true;}
static double reproj(const Camera&P,const std::vector<std::array<double,3>>&pts,const std::vector<Point>&pix){double s=0;for(size_t i=0;i<pts.size();i++){auto a=pts[i];double w=P[8]*a[0]+P[9]*a[1]+P[10]*a[2]+P[11],u=(P[0]*a[0]+P[1]*a[1]+P[2]*a[2]+P[3])/w,v=(P[4]*a[0]+P[5]*a[1]+P[6]*a[2]+P[7])/w;s+=dist({u,v},pix[i]);}return s/pts.size();}
static uint64_t packVoxel(int x,int y,int z){
 return ((uint64_t(x+1024)&0xFFFFFULL)<<40)|((uint64_t(y+1024)&0xFFFFFULL)<<20)|(uint64_t(z+1024)&0xFFFFFULL);
}
static bool refineCameraWithOffPlaneSeeds(const Context&c,const Plane&plane,const std::vector<Block>&planeBlocks,const std::vector<Block>&seeds,Camera&P){
 if(planeBlocks.size()<3||seeds.empty())return false;
 H3 planeInv;if(!inv3(plane.H,planeInv))return false;
 std::vector<std::array<double,3>>p3;
 std::vector<Point>pix;
 for(const auto&b:planeBlocks)for(int i=0;i<4;i++){p3.push_back({b.x+unit[i].x,0.0,b.z+unit[i].y});pix.push_back(b.quad[i]);}
 int added=0;
 for(const auto&s:seeds){
  Quad loc;for(int i=0;i<4;i++)loc[i]=applyH(planeInv,s.quad[i].x,s.quad[i].y);
  double ox=std::floor(loc[0].x+.5),oz=std::floor(loc[0].y+.5),onPlaneErr=0;
  for(int i=0;i<4;i++)onPlaneErr=std::max(onPlaneErr,dist(loc[i],unit[i]+Point(ox,oz)));
  if(onPlaneErr<0.25)continue;
  double bestErr=INFINITY,secondErr=INFINITY;
  int bx=0,by=0,bz=0;
  for(int y=-12;y<=12;y++){
   if(y==0)continue;
   H3 Hy={P[0],P[2],P[3]+P[1]*y,P[4],P[6],P[7]+P[5]*y,P[8],P[10],P[11]+P[9]*y},Iy;
   if(!inv3(Hy,Iy))continue;
   Point p0=applyH(Iy,s.quad[0].x,s.quad[0].y);
   if(!std::isfinite(p0.x)||!std::isfinite(p0.y))continue;
   int x0=int(std::llround(p0.x)),z0=int(std::llround(p0.y));
   for(int dz=-1;dz<=1;dz++)for(int dx=-1;dx<=1;dx++){
    int x=x0+dx,z=z0+dz;
    if(x<-30||x>30||z<-30||z>30)continue;
    Quad q;if(!projectQuad(P,x,y,z,c.w,c.h,q))continue;
    double err=0;for(int i=0;i<4;i++)err+=dist(q[i],s.quad[i]);
    err*=0.25;
    if(err<bestErr){secondErr=bestErr;bestErr=err;bx=x;by=y;bz=z;}
    else if(err<secondErr)secondErr=err;
   }
  }
  double scale=minSide(s.quad);
  if(bestErr<std::min(15.0,scale*0.25)&&bestErr<secondErr*0.55){
   for(int i=0;i<4;i++){p3.push_back({bx+unit[i].x,double(by),bz+unit[i].y});pix.push_back(s.quad[i]);}
   added++;
  }
 }
 if(!added)return false;
 Camera P2;
 if(!dlt(p3,pix,P2)||!(reproj(P2,p3,pix)<2.5))return false;
 P=P2;
 return true;
}
static std::vector<Block>voxel(const Context&c,const Plane&plane,const std::vector<Block>&planeBlocks,const std::vector<Block>&seeds,const Refiner&r,Camera&bestP,bool&has){
 struct CachedVoxel{Quad projQ;Block verified;bool accepted;};
 Camera P;double f;if(!cameraFromPlane(plane.H,c.w,c.h,P,f))return {};
 refineCameraWithOffPlaneSeeds(c,plane,planeBlocks,seeds,P);
 int prev=-1;std::vector<Block>best;has=false;
 std::unordered_map<uint64_t,CachedVoxel>roundCache;
 for(int round=0;round<4;round++){
  std::vector<Block>cand,accept,kept;std::vector<Quad>qs;
  std::unordered_map<uint64_t,CachedVoxel>nextCache;
  for(int y=-12;y<=12;y++)for(int z=-30;z<=30;z++)for(int x=-30;x<=30;x++){
   Quad q;if(!projectQuad(P,x,y,z,c.w,c.h,q))continue;
   uint64_t key=packVoxel(x,y,z);
   auto it=roundCache.find(key);
   if(it!=roundCache.end()){
    double drift=0;
    for(int i=0;i<4;i++)drift=std::max(drift,dist(q[i],it->second.projQ[i]));
    if(drift<0.15){
     nextCache.emplace(key,it->second);
     if(it->second.accepted)accept.push_back(it->second.verified);
     continue;
    }
   }
   Block b;b.quad=q;b.x=x;b.y=y;b.z=z;cand.push_back(b);qs.push_back(q);
  }
  auto vs=qs.empty()?std::vector<Verdict>{}:classifyBatchNative(c,qs);
  for(size_t i=0;i<cand.size();i++){
   auto b=verify(c,cand[i].quad,r,vs[i],true);
   b.x=cand[i].x;b.y=cand[i].y;b.z=cand[i].z;
   bool ok=b.valid&&b.score>c.opts.relaxed&&b.margin>c.opts.margin&&b.votes==4;
   nextCache.emplace(packVoxel(b.x,b.y,b.z),CachedVoxel{cand[i].quad,b,ok});
   if(ok)accept.push_back(b);
  }
  roundCache=std::move(nextCache);
  std::stable_sort(accept.begin(),accept.end(),[](auto&a,auto&b){return a.score>b.score;});
  std::vector<Point>centers;std::vector<double>sides;
  for(auto&b:accept){
   Point center{};for(auto p:b.quad){center.x+=p.x/4;center.y+=p.y/4;}
   double sd=minSide(b.quad);bool overlap=false;
   for(size_t i=0;i<kept.size();i++)if(dist(centers[i],center)<.5*std::min(sd,sides[i])){overlap=true;break;}
   if(!overlap){kept.push_back(b);centers.push_back(center);sides.push_back(sd);}
  }
  std::set<int>heights;for(auto&b:kept)heights.insert(b.y);
  if(!has||kept.size()>best.size()){best=kept;bestP=P;has=true;}
  if(int(kept.size())<=prev||kept.size()<size_t(c.opts.refitFlat&&heights.size()==1?3:6))break;
  prev=kept.size();
  if(heights.size()==1){
   if(!c.opts.refitFlat)break;
   std::vector<Point>src,dst;for(auto&b:kept)for(int i=0;i<4;i++){src.push_back(unit[i]+Point(b.x,b.z));dst.push_back(b.quad[i]);}
   H3 H;Camera flat;if(!homography(src,dst,H)||!cameraFromPlane(H,c.w,c.h,flat,f))break;
   P=flat;for(int i:{0,4,8})P[i+3]-=P[i+1]*(*heights.begin());
   continue;
  }
  std::vector<std::array<double,3>>p3;std::vector<Point>pix;
  for(auto&b:kept)for(int i=0;i<4;i++){p3.push_back({b.x+unit[i].x,double(b.y),b.z+unit[i].y});pix.push_back(b.quad[i]);}
  Camera p2;if(!dlt(p3,pix,p2)||!(reproj(p2,p3,pix)<3))break;
  P=p2;
 }
 return best;
}
static std::vector<Block>topDown(const Context&c){std::vector<double>periods;const uint8_t*g=c.g.data();for(int axis=0;axis<2;axis++){int n=axis?c.h:c.w,other=axis?c.w:c.h;std::vector<double>e(n-1);for(int j=0;j<other;j+=3)for(int i=0;i<n-1;i++){int p=axis?i*c.w+j:j*c.w+i;e[i]+=std::abs(int(g[p+(axis?c.w:1)])-int(g[p]));}double mean=std::accumulate(e.begin(),e.end(),0.)/e.size();std::vector<std::pair<double,double>>pk;for(int k=std::ceil(n/24.);k<n/2.;k++){double re=0,im=0,an=2*std::acos(-1)*k/e.size();for(size_t i=0;i<e.size();i++){double v=e[i]-mean;re+=v*std::cos(an*i);im+=v*std::sin(an*i);}pk.push_back({e.size()/double(k)*16,re*re+im*im});}std::stable_sort(pk.begin(),pk.end(),[](auto&a,auto&b){return a.second>b.second;});for(size_t i=0;i<std::min(size_t(2),pk.size());i++)periods.push_back(pk[i].first);}std::vector<int>sizes;for(double p:periods)for(int d:{-1,0,1}){int n=std::floor(p+.5)+d;if(n>=48&&n<std::min(c.w,c.h)*.6&&std::find(sizes.begin(),sizes.end(),n)==sizes.end())sizes.push_back(n);}Block best;Quad bestQuad{};bool hasBest=false;double bestScore=.72;for(int n:sizes){int st=std::max(2,n/32);std::vector<Quad>batch;for(int y=0;y<c.h-n;y+=st)for(int x=0;x<c.w-n;x+=st)batch.push_back({Point(x,y),Point(x+n,y),Point(x+n,y+n),Point(x,y+n)});if(batch.empty())continue;auto vs=classifyBatchNative(c,batch);for(size_t i=0;i<batch.size();i++)if(vs[i].valid&&vs[i].score>bestScore){bestScore=vs[i].score;bestQuad=batch[i];hasBest=true;}}if(hasBest){auto r=classify(c,bestQuad);if(r.valid&&r.score>.72){static_cast<Verdict&>(best)=r;best.quad=bestQuad;}}return best.valid?std::vector<Block>{best}:std::vector<Block>{};}
Result runDetection(Context&c,const std::vector<Quad>&raw,bool manual,Timings&t,const Refiner*sharedRefine){
 Result result;
 Refiner baseRefine=(sharedRefine&&*sharedRefine)?*sharedRefine:makeEccRefiner(c);
 Refiner refine=[&](const Quad&q,int rot,Quad&out){double start=clockMs();bool ok=baseRefine(q,rot,out);t.ecc+=clockMs()-start;return ok;};double start=clockMs();for(auto&q:raw){auto b=verify(c,q,refine,{});if(b.valid&&b.score>=c.opts.accept&&b.margin>=c.opts.margin&&b.votes==4)result.seeds.push_back(b);}if(result.seeds.empty()&&!manual){double fs=clockMs();auto seeds=topDown(c);t.latticeFallback+=clockMs()-fs;for(auto b:seeds)if(b.score>=c.opts.accept)result.seeds.push_back(b);}t.verify+=clockMs()-start;if(result.seeds.empty()){result.reason=manual?"your quad did not verify as a grass top (correlation too low). Make the corners tighter.":"no reliable grass tile found automatically. Draw one tile quad by hand.";return result;}start=clockMs();Plane plane=planeFromSeeds(result.seeds);if(!plane.valid){result.reason="seeds do not share a plane lattice";return result;}auto blocks=growPlane(c,plane.H,refine,28);result.mode="planar lattice (all y=0)";if(c.opts.refitPlane)for(int round=0;round<3&&blocks.size()>=3;round++){std::vector<Point>src,dst;for(auto&b:blocks)for(int i=0;i<4;i++){src.push_back(unit[i]+Point(b.x,b.z));dst.push_back(b.quad[i]);}H3 H;if(!homography(src,dst,H))break;auto next=growPlane(c,H,refine,28);if(next.size()<=blocks.size())break;blocks=next;plane.H=H;}t.plane+=clockMs()-start;start=clockMs();bool has=false;Camera cam;auto v=voxel(c,plane,blocks,result.seeds,refine,cam,has);if(has&&v.size()>=blocks.size()){blocks=v;result.camera=cam;result.hasCamera=true;std::set<int>heights;for(auto&b:blocks)heights.insert(b.y);result.mode="3D voxel lattice (heights ";for(int h:heights){if(result.mode.back()!=' ')result.mode+=",";result.mode+=std::to_string(h);}result.mode+=")";}t.voxel+=clockMs()-start;std::stable_sort(blocks.begin(),blocks.end(),[](auto&a,auto&b){return a.score>b.score;});result.blocks=blocks;result.planeH=plane.H;result.hasPlane=true;result.reason=blocks.empty()?"seed found but no neighbours verified":"";return result;}
std::vector<Block>mergeAligned(const std::vector<Block>&a,const std::vector<Block>&incoming){using Key=std::array<int,3>;std::vector<Key>evidence,order;std::map<Key,int>votes;for(auto&b:incoming){const Block*nearest=nullptr;double best=INFINITY;for(auto&o:a){double scale=minSide(o.quad),err=0;for(int i=0;i<4;i++)err+=dist(b.quad[i],o.quad[i]);err/=4;if(err<std::max(3.,scale*.12)&&err<best){best=err;nearest=&o;}}if(!nearest)continue;if(nearest->rotation!=b.rotation)return a;Key key={nearest->x-b.x,nearest->y-b.y,nearest->z-b.z};evidence.push_back(key);if(!votes.count(key))order.push_back(key);votes[key]++;}if(order.empty())return a;Key winner=order[0];for(auto key:order)if(votes[key]>votes[winner])winner=key;int minVotes=std::min<int>(3,int(a.size()));if(a.size()>=3){if(votes[winner]<3||votes[winner]<evidence.size()*.9)return a;}else if(votes[winner]<minVotes||votes[winner]<int(evidence.size()))return a;auto merged=a;std::map<Key,int>coords;for(auto&b:a)coords[{b.x,b.y,b.z}]=b.rotation;for(auto&b:incoming){Key key={b.x+winner[0],b.y+winner[1],b.z+winner[2]};if(coords.count(key)&&coords[key]!=b.rotation)return a;}for(auto b:incoming){b.x+=winner[0];b.y+=winner[1];b.z+=winner[2];Key key={b.x,b.y,b.z};if(!coords.count(key)){merged.push_back(b);coords[key]=b.rotation;}}return merged;}
}
