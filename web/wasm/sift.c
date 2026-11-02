#include "vl/sift.h"
#include <stdlib.h>
static float out[20000*130];
float *get_out(){return out;}
int extract(float *im,int w,int h){
 VlSiftFilt *f=vl_sift_new(w,h,-1,3,-1);vl_sift_set_peak_thresh(f,1.7);vl_sift_set_edge_thresh(f,15);int n=0;
 for(int err=vl_sift_process_first_octave(f,im);!err;err=vl_sift_process_next_octave(f)){
 vl_sift_detect(f);const VlSiftKeypoint *k=vl_sift_get_keypoints(f);int count=vl_sift_get_nkeypoints(f);
 for(int i=0;i<count&&n<20000;i++){double angles[4];int na=vl_sift_calc_keypoint_orientations(f,angles,k+i);for(int a=0;a<na&&n<20000;a++){out[n*130]=k[i].x;out[n*130+1]=k[i].y;vl_sift_calc_keypoint_descriptor(f,out+n*130+2,k+i,angles[a]);n++;}}
 }vl_sift_delete(f);return n;
}
