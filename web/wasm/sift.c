#include "vl/sift.h"
#include "vl/generic.h"
#include <stdlib.h>
#include <stdint.h>
#include <setjmp.h>

/* o_min=-1 uses 22 float planes at twice each input dimension:
 * 352 bytes/input pixel, plus input, keypoints and the 10.4 MB output.
 * Reject oversized direct callers; JS downsamples and maps coordinates back. */
#define MAX_PIXELS 1048576
#define MAX_EDGE 2048
#define MAX_ALLOCS 64
static float out[20000*130];
static void *allocs[MAX_ALLOCS];
static jmp_buf oom;
static void remember(void *p) {
 if (!p) longjmp(oom,1);
 for(int i=0;i<MAX_ALLOCS;i++) if(!allocs[i]) {allocs[i]=p;return;}
 free(p);longjmp(oom,1);
}
static void *checked_malloc(size_t n) {void *p=malloc(n);remember(p);return p;}
static void checked_free(void *p) {
 for(int i=0;i<MAX_ALLOCS;i++) if(allocs[i]==p) {allocs[i]=NULL;break;}
 free(p);
}
static void *checked_calloc(size_t n,size_t s) {void *p=calloc(n,s);remember(p);return p;}
static void *checked_realloc(void *p,size_t n) {
 if(!p)return checked_malloc(n);
 for(int i=0;i<MAX_ALLOCS;i++) if(allocs[i]==p) {
  void *q=realloc(p,n);if(!q)longjmp(oom,1);allocs[i]=q;return q;
 }
 longjmp(oom,1);
}
float *get_out(){return out;}
int extract(float *im,int w,int h){
 if(!im||w<8||h<8||w>MAX_EDGE||h>MAX_EDGE||(uint64_t)w*h>MAX_PIXELS)return -1;
 vl_set_alloc_func(checked_malloc,checked_realloc,checked_calloc,checked_free);
 if(setjmp(oom)) {
  for(int i=0;i<MAX_ALLOCS;i++){free(allocs[i]);allocs[i]=NULL;}
  vl_set_alloc_func(malloc,realloc,calloc,free);
  return -2;
 }
 VlSiftFilt *f=vl_sift_new(w,h,-1,3,-1);vl_sift_set_peak_thresh(f,1.7);vl_sift_set_edge_thresh(f,15);int n=0;
 for(int err=vl_sift_process_first_octave(f,im);!err;err=vl_sift_process_next_octave(f)){
 vl_sift_detect(f);const VlSiftKeypoint *k=vl_sift_get_keypoints(f);int count=vl_sift_get_nkeypoints(f);
 for(int i=0;i<count&&n<20000;i++){double angles[4];int na=vl_sift_calc_keypoint_orientations(f,angles,k+i);for(int a=0;a<na&&n<20000;a++){out[n*130]=k[i].x;out[n*130+1]=k[i].y;vl_sift_calc_keypoint_descriptor(f,out+n*130+2,k+i,angles[a]);n++;}}
 }vl_sift_delete(f);
 vl_set_alloc_func(malloc,realloc,calloc,free);
 return n;
}
