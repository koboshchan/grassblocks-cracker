/** @file sift.c
 ** @brief SIFT - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page sift Scale Invariant Feature Transform (SIFT)
@author Andrea Vedaldi
@par "Credits:" May people have contributed with suggestions and bug
reports. Although the following list is certainly incomplete, we would
like to thank: Wei Dong, Loic, Giuseppe, Liu, Erwin, P. Ivanov, and
Q. S. Luo.
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref sift.h implements a @ref sift-usage "SIFT filter object", a
reusable object to extract SIFT features @cite{lowe99object} from one
or multiple images.

This is the original VLFeat implementation of SIFT, designed to be
compatible with Lowe's original SIFT. See @ref covdet for a different
version of SIFT integrated in the more general covariant feature
detector engine.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section sift-intro Overview
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A SIFT feature is a selected image region (also called keypoint) with
an associated descriptor. Keypoints are extracted by the <b>@ref
sift-intro-detector "SIFT detector"</b> and their descriptors are
computed by the <b>@ref sift-intro-descriptor "SIFT descriptor"</b>.  It is
also common to use independently the SIFT detector (i.e. computing the
keypoints without descriptors) or the SIFT descriptor (i.e.  computing
descriptors of custom keypoints).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-intro-detector SIFT detector
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A SIFT <em>keypoint</em> is a circular image region with an
orientation. It is described by a geometric <em>frame</em> of four
parameters: the keypoint center coordinates @e x and @e y, its @e
scale (the radius of the region), and its @e orientation (an angle
expressed in radians). The SIFT detector uses as keypoints image
structures which resemble &ldquo;blobs&rdquo;. By searching for blobs
at multiple scales and positions, the SIFT detector is invariant (or,
more accurately, covariant) to translation, rotations, and re scaling
of the image.

The keypoint orientation is also determined from the local image
appearance and is covariant to image rotations. Depending on the
symmetry of the keypoint appearance, determining the orientation can
be ambiguous. In this case, the SIFT detectors returns a list of up to
four possible orientations, constructing up to four frames (differing
only by their orientation) for each detected image blob.

@image html sift-frame.png "SIFT keypoints are circular image regions with an orientation."

There are several parameters that influence the detection of SIFT
keypoints. First, searching keypoints at multiple scales is obtained
by constructing a so-called &ldquo;Gaussian scale space&rdquo;. The
scale space is just a collection of images obtained by progressively
smoothing the input image, which is analogous to gradually reducing
the image resolution. Conventionally, the smoothing level is called
<em>scale</em> of the image. The construction of the scale space is
influenced by the following parameters, set when creating the SIFT
filter object by ::vl_sift_new():

- <b>Number of octaves</b>. Increasing the scale by an octave means
  doubling the size of the smoothing kernel, whose effect is roughly
  equivalent to halving the image resolution. By default, the scale
  space spans as many octaves as possible (i.e. roughly <code>
  log2(min(width,height)</code>), which has the effect of searching
  keypoints of all possible sizes.
- <b>First octave index</b>. By convention, the octave of index 0
  starts with the image full resolution. Specifying an index greater
  than 0 starts the scale space at a lower resolution (e.g. 1 halves
  the resolution). Similarly, specifying a negative index starts the
  scale space at an higher resolution image, and can be useful to
  extract very small features (since this is obtained by interpolating
  the input image, it does not make much sense to go past -1).
- <b>Number of levels per octave</b>. Each octave is sampled at this
  given number of intermediate scales (by default 3). Increasing this
  number might in principle return more refined keypoints, but in
  practice can make their selection unstable due to noise (see [1]).

Keypoints are further refined by eliminating those that are likely to
be unstable, either because they are selected nearby an image edge,
rather than an image blob, or are found on image structures with low
contrast. Filtering is controlled by the follow:

- <b>Peak threshold.</b> This is the minimum amount of contrast to
  accept a keypoint. It is set by configuring the SIFT filter object
  by ::vl_sift_set_peak_thresh().
- <b>Edge threshold.</b> This is the edge rejection threshold. It is
  set by configuring the SIFT filter object by
  ::vl_sift_set_edge_thresh().

<table>
 <caption>Summary of the parameters influencing the SIFT detector.</caption>
 <tr style="font-weight:bold;">
 <td>Parameter</td>
 <td>See also</td>
 <td>Controlled by</td>
 <td>Comment</td>
 </tr>
 <tr>
 <td>number of octaves</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_new</td>
 <td></td>
 </tr>
 <tr>
 <td>first octave index</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_new</td>
 <td>set to -1 to extract very small features</td>
 </tr>
 <tr>
 <td>number of scale levels per octave</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_new</td>
 <td>can affect the number of extracted keypoints</td>
 </tr>
 <tr>
 <td>edge threshold</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_set_edge_thresh</td>
 <td>decrease to eliminate more keypoints</td>
 </tr>
 <tr>
 <td>peak threshold</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_set_peak_thresh</td>
 <td>increase to eliminate more keypoints</td>
 </tr>
</table>

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-intro-descriptor SIFT Descriptor
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@sa @ref sift-tech-descriptor "Descriptor technical details"

A SIFT descriptor is a 3-D spatial histogram of the image gradients in
characterizing the appearance of a keypoint. The gradient at each
pixel is regarded as a sample of a three-dimensional elementary
feature vector, formed by the pixel location and the gradient
orientation. Samples are weighed by the gradient norm and accumulated
in a 3-D histogram @em h, which (up to normalization and clamping)
forms the SIFT descriptor of the region. An additional Gaussian
weighting function is applied to give less importance to gradients
farther away from the keypoint center. Orientations are quantized into
eight bins and the spatial coordinates into four each, as follows:

@image html sift-descr-easy.png "The SIFT descriptor is a spatial histogram of the image gradient."

SIFT descriptors are computed by either calling
::vl_sift_calc_keypoint_descriptor or
::vl_sift_calc_raw_descriptor. They accept as input a keypoint
frame, which specifies the descriptor center, its size, and its
orientation on the image plane. The following parameters influence the
descriptor calculation:

- <b>magnification factor</b>. The descriptor size is determined by
multiplying the keypoint scale by this factor. It is set by
::vl_sift_set_magnif.
- <b>Gaussian window size</b>. The descriptor support is determined by
a Gaussian window, which discounts gradient contributions farther away
from the descriptor center. The standard deviation of this window is
set by ::vl_sift_set_window_size and expressed in unit of bins.

VLFeat SIFT descriptor uses the following convention. The @em y axis
points downwards and angles are measured clockwise (to be consistent
with the standard image convention). The 3-D histogram (consisting of
@f$ 8 \times 4 \times 4 = 128 @f$ bins) is stacked as a single
128-dimensional vector, where the fastest varying dimension is the
orientation and the slowest the @em y spatial coordinate. This is
illustrated by the following figure.

@image html sift-conv-vlfeat.png "VLFeat conventions"

@note Keypoints (frames) D. Lowe's SIFT implementation convention is
slightly different: The @em y axis points upwards and the angles are
measured counter-clockwise.

@image html sift-conv.png "D. Lowes' SIFT implementation conventions"

<table>
 <caption>Summary of the parameters influencing the SIFT descriptor.</caption>
 <tr style="font-weight:bold;">
 <td>Parameter</td>
 <td>See also</td>
 <td>Controlled by</td>
 <td>Comment</td>
 </tr>
 <tr>
 <td>magnification factor</td>
 <td> @ref sift-intro-descriptor </td>
 <td>::vl_sift_set_magnif</td>
 <td>increase this value to enlarge the image region described</td>
 </tr>
 <tr>
 <td>Gaussian window size</td>
 <td> @ref sift-intro-descriptor </td>
 <td>::vl_sift_set_window_size</td>
 <td>smaller values let the center of the descriptor count more</td>
 </tr>
</table>


<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section sift-intro-extensions Extensions
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<b>Eliminating low-contrast descriptors.</b> Near-uniform patches do
not yield stable keypoints or descriptors. ::vl_sift_set_norm_thresh()
can be used to set a threshold on the average norm of the local
gradient to zero-out descriptors that correspond to very low contrast
regions. By default, the threshold is equal to zero, which means that
no descriptor is zeroed. Normally this option is useful only with
custom keypoints, as detected keypoints are implicitly selected at
high contrast image regions.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section sift-usage Using the SIFT filter object
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The code provided in this module can be used in different ways.  You
can instantiate and use a <b>SIFT filter</b> to extract both SIFT
keypoints and descriptors from one or multiple images. Alternatively,
you can use one of the low level functions to run only a part of the
SIFT algorithm (for instance, to compute the SIFT descriptors of
custom keypoints).

To use a <b>SIFT filter</b> object:

- Initialize a SIFT filter object with ::vl_sift_new(). The filter can
  be reused for multiple images of the same size (e.g. for an entire
  video sequence).
- For each octave in the scale space:
  - Compute the next octave of the DOG scale space using either
   ::vl_sift_process_first_octave() or ::vl_sift_process_next_octave()
   (stop processing if ::VL_ERR_EOF is returned).
  - Run the SIFT detector with ::vl_sift_detect() to get the keypoints.
  - For each keypoint:
    - Use ::vl_sift_calc_keypoint_orientations() to get the keypoint orientation(s).
    - For each orientation:
      - Use ::vl_sift_calc_keypoint_descriptor() to get the keypoint descriptor.
- Delete the SIFT filter by ::vl_sift_delete().

To compute SIFT descriptors of custom keypoints, use
::vl_sift_calc_raw_descriptor().

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section sift-tech Technical details
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-tech-ss Scale space
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

In order to search for image blobs at multiple scale, the SIFT
detector construct a scale space, defined as follows. Let
@f$I_0(\mathbf{x})@f$ denote an idealized <em>infinite resolution</em>
image. Consider the  <em>Gaussian kernel</em>

@f[
 g_{\sigma}(\mathbf{x})
 =
 \frac{1}{2\pi\sigma^2}
 \exp
 \left(
 -\frac{1}{2}
 \frac{\mathbf{x}^\top\mathbf{x}}{\sigma^2}
 \right)
@f]

The <b>Gaussian scale space</b> is the collection of smoothed images

@f[
 I_\sigma = g_\sigma * I,  \quad \sigma \geq 0.
@f]

The image at infinite resolution @f$ I_0 @f$ is useful conceptually,
but is not available to us; instead, the input image @f$ I_{\sigma_n}
@f$ is assumed to be pre-smoothed at a nominal level @f$ \sigma_n =
0.5 @f$ to account for the finite resolution of the pixels. Thus in
practice the scale space is computed by

@f[
I_\sigma = g_{\sqrt{\sigma^2 - \sigma_n^2}} * I_{\sigma_n},
\quad \sigma \geq \sigma_n.
@f]

Scales are sampled at logarithmic steps given by

@f[
\sigma = \sigma_0 2^{o+s/S},
\quad s = 0,\dots,S-1,
\quad o = o_{\min}, \dots, o_{\min}+O-1,
@f]

where @f$ \sigma_0 = 1.6 @f$ is the <em>base scale</em>, @f$ o_{\min}
@f$ is the <em>first octave index</em>, @em O the <em>number of
octaves</em> and @em S the <em>number of scales per octave</em>.

Blobs are detected as local extrema of the <b>Difference of
Gaussians</b> (DoG) scale space, obtained by subtracting successive
scales of the Gaussian scale space:

@f[
\mathrm{DoG}_{\sigma(o,s)} = I_{\sigma(o,s+1)} - I_{\sigma(o,s)}
@f]

At each next octave, the resolution of the images is halved to save
computations. The images composing the Gaussian and DoG scale space
can then be arranged as in the following figure:

@image html sift-ss.png  "GSS and DoG scale space  structures."

The black vertical segments represent images of the Gaussian Scale
Space (GSS), arranged by increasing scale @f$\sigma@f$.  Notice that
the scale level index @e s varies in a slightly redundant set

@f[
s = -1, \dots, S+2
@f]

This simplifies glueing together different octaves and extracting DoG
maxima (required by the SIFT detector).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-tech-detector Detector
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The SIFT frames (keypoints) are extracted based on local extrema
(peaks) of the DoG scale space. Numerically, local extrema are
elements whose @f$ 3 \times 3 \times 3 @f$ neighbors (in space and
scale) have all smaller (or larger) value.  Once extracted, local
extrema are quadratically interpolated (this is very important
especially at the lower resolution scales in order to have accurate
keypoint localization at the full resolution).  Finally, they are
filtered to eliminate low-contrast responses or responses close to
edges and the orientation(s) are assigned, as explained next.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-detector-peak Eliminating low contrast responses
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Peaks which are too short may have been generated by noise and are
discarded.  This is done by comparing the absolute value of the DoG
scale space at the peak with the <b>peak threshold</b> @f$t_p@f$ and
discarding the peak its value is below the threshold.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-detector-edge Eliminating edge responses
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Peaks which are too flat are often generated by edges and do not yield
stable features. These peaks are detected and removed as follows.
Given a peak @f$x,y,\sigma@f$, the algorithm evaluates the @em x,@em y
Hessian of of the DoG scale space at the scale @f$\sigma@f$.  Then the
following score (similar to the Harris function) is computed:

@f[
\frac{(\mathrm{tr}\,D(x,y,\sigma))^2}{\det D(x,y,\sigma)},
\quad
D =
\left[
\begin{array}{cc}
\frac{\partial^2 \mathrm{DoG}}{\partial x^2} &
\frac{\partial^2 \mathrm{DoG}}{\partial x\partial y} \\
\frac{\partial^2 \mathrm{DoG}}{\partial x\partial y} &
\frac{\partial^2 \mathrm{DoG}}{\partial y^2}
\end{array}
\right].
 @f]

This score has a minimum (equal to 4) when both eigenvalues of the
Jacobian are equal (curved peak) and increases as one of the
eigenvalues grows and the other stays small. Peaks are retained if the
score is below the quantity @f$(t_e+1)(t_e+1)/t_e@f$, where @f$t_e@f$
is the <b>edge threshold</b>. Notice that this quantity has a minimum
equal to 4 when @f$t_e=1@f$ and grows thereafter. Therefore the range
of the edge threshold is @f$[1,\infty)@f$.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-tech-detector-orientation Orientation assignment
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A peak in the DoG scale space fixes 2 parameters of the keypoint: the
position and scale. It remains to choose an orientation. In order to
do this, SIFT computes an histogram of the gradient orientations in a
Gaussian window with a standard deviation which is 1.5 times bigger
than the scale @f$\sigma@f$ of the keypoint.

@image html sift-orient.png

This histogram is then smoothed and the maximum is selected.  In
addition to the biggest mode, up to other three modes whose amplitude
is within the 80% of the biggest mode are retained and returned as
additional orientations.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-tech-descriptor Descriptor
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A SIFT descriptor of a local region (keypoint) is a 3-D spatial
histogram of the image gradients.  The gradient at each pixel is
regarded as a sample of a three-dimensional elementary feature vector,
formed by the pixel location and the gradient orientation. Samples are
weighed by the gradient norm and accumulated in a 3-D histogram @em h,
which (up to normalization and clamping) forms the SIFT descriptor of
the region. An additional Gaussian weighting function is applied to
give less importance to gradients farther away from the keypoint
center.

<!-- @image html sift-bins.png -->

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-descriptor-can Construction in the canonical frame
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Denote the gradient vector field computed at the scale @f$ \sigma @f$ by
@f[
  J(x,y) = \nabla I_\sigma(x,y)
  =
  \left[\begin{array}{cc}
  \frac{\partial I_\sigma}{\partial x} &
  \frac{\partial I_\sigma}{\partial y} &
  \end{array}\right]
@f]

The descriptor is a 3-D spatial histogram capturing the distribution
of @f$ J(x,y) @f$. It is convenient to describe its construction in
the <em>canonical frame</em>. In this frame, the image and descriptor
axes coincide and each spatial bin has side 1. The histogram has @f$
N_\theta \times N_x \times N_y @f$ bins (usually @f$ 8 \times 4 \times
4 @f$), as in the following figure:

@image html sift-can.png Canonical SIFT descriptor and spatial binning functions

Bins are indexed by a triplet of indexes <em>t, i, j</em> and their
centers are given by

@f{eqnarray*}
 \theta_t &=& \frac{2\pi}{N_\theta} t, \quad t = 0,\dots,N_{\theta}-1, \\
 x_i &=& i - \frac{N_x -1}{2}, \quad i = 0,\dots,N_x-1, \\
 y_j &=& j - \frac{N_x -1}{2}, \quad j = 0,\dots,N_y-1. \\
@f}

The histogram is computed by using trilinear interpolation, i.e.  by
weighing contributions by the <em>binning functions</em>

@f{eqnarray*}
  \displaystyle
  w(z) &=& \mathrm{max}(0, 1 - |z|),
  \\
  \displaystyle
  w_\mathrm{ang}(z) &=& \sum_{k=-\infty}^{+\infty}
  w\left(
  \frac{N_\theta}{2\pi} z + N_\theta k
  \right).
@f}

The gradient vector field is transformed in a three-dimensional
density map of weighed contributions

@f[
   f(\theta, x, y) = |J(x,y)| \delta(\theta - \angle J(x,y))
@f]

The historam is localized in the keypoint support by a Gaussian window
of standard deviation @f$ \sigma_{\mathrm{win}} @f$. The histogram is
then given by

@f{eqnarray*}
 h(t,i,j) &=& \int
 g_{\sigma_\mathrm{win}}(x,y)
 w_\mathrm{ang}(\theta - \theta_t) w(x-x_i) w(y-y_j)
 f(\theta,x,y)
 d\theta\,dx\,dy
\\
&=& \int
 g_{\sigma_\mathrm{win}}(x,y)
 w_\mathrm{ang}(\angle J(x,y) - \theta_t) w(x-x_i) w(y-y_j)
 |J(x,y)|\,dx\,dy
@f}

In post processing, the histogram is @f$ l^2 @f$ normalized, then
clamped at 0.2, and @f$ l^2 @f$ normalized again.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-descriptor-image Calculation in the image frame
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Invariance to similarity transformation is attained by attaching
descriptors to SIFT keypoints (or other similarity-covariant frames).
Then projecting the image in the canonical descriptor frames has the
effect of undoing the image deformation.

In practice, however, it is convenient to compute the descriptor
directly in the image frame. To do this, denote with a hat quantities
relative to the canonical frame and without a hat quantities relative
to the image frame (so for instance @f$ \hat x @f$ is the @e
x-coordinate in the canonical frame and @f$ x @f$ the x-coordinate in
the image frame). Assume that canonical and image frame are
related by an affinity:

@f[
  \mathbf{x} = A \hat{\mathbf{x}} + T,
  \qquad
  \mathbf{x} =
  \begin{bmatrix}{c}
    x \\
    y
  \end{bmatrix},
  \quad
  \mathbf{x} =
  \begin{bmatrix}{c}
    \hat x \\
    \hat y
  \end{bmatrix}.
@f]

@image html sift-image-frame.png

Then all quantities can be computed in the image frame directly. For instance,
the image at infinite resolution in the two frames are related by

@f[
 \hat I_0(\hat{\mathbf{x}})  = I_0(\mathbf{x}),
 \qquad
 \mathbf{x} = A \hat{\mathbf{x}} + T.
@f]

The canonized image at scale @f$ \hat \sigma @f$ is in relation with the scaled image

@f[
 \hat I_{\hat{\sigma}}(\hat{\mathbf{x}})  = I_{A\hat{\sigma}}(\mathbf{x}),
 \qquad \mathbf{x} = A \hat{\mathbf{x}} + T
@f]

where, by generalizing the previous definitions, we have

@f[
 I_{A\hat \sigma}(\mathbf{x}) = (g_{A\hat\sigma} * I_0)(\mathbf{x}),
\quad
 g_{A\hat\sigma}(\mathbf{x})
 =
 \frac{1}{2\pi|A|\hat \sigma^2}
 \exp
 \left(
 -\frac{1}{2}
 \frac{\mathbf{x}^\top A^{-\top}A^{-1}\mathbf{x}}{\hat \sigma^2}
 \right)
@f]

Deriving shows that the gradient fields are in relation

@f[
  \hat J(\hat{\mathbf{x}}) = J(\mathbf{x}) A,
 \quad J(\mathbf{x}) = (\nabla I_{A\hat\sigma})(\mathbf{x}),
 \qquad \mathbf{x} = A \hat{\mathbf{x}} + T.
@f]

Therefore we can compute the descriptor either in the image or canonical frame as:

@f{eqnarray*}
 h(t,i,j)
 &=&
 \int
 g_{\hat \sigma_\mathrm{win}}(\hat{\mathbf{x}})\,
 w_\mathrm{ang}(\angle \hat J(\hat{\mathbf{x}}) - \theta_t)\,
 w_{ij}(\hat{\mathbf{x}})\,
 |\hat J(\hat{\mathbf{x}})|\,
 d\hat{\mathbf{x}}
 \\
 &=& \int
 g_{A \hat \sigma_\mathrm{win}}(\mathbf{x} - T)\,
 w_\mathrm{ang}(\angle J(\mathbf{x})A - \theta_t)\,
 w_{ij}(A^{-1}(\mathbf{x} - T))\,
 |J(\mathbf{x})A|\,
 d\mathbf{x}.
@f}

where we defined the product of the two spatial binning functions

@f[
 w_{ij}(\hat{\mathbf{x}}) = w(\hat x - \hat x_i) w(\hat y - \hat y_j)
@f]


In the actual implementation, this integral is computed by visiting a
rectangular area of the image that fully contains the keypoint grid
(along with half a bin border to fully include the bin windowing
function). Since the descriptor can be rotated, this area is a
rectangle of sides @f$m/2\sqrt{2} (N_x+1,N_y+1)@f$ (see also the
illustration).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-descriptor-std Standard SIFT descriptor
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

For a SIFT-detected keypoint of center @f$ T @f$, scale @f$ \sigma @f$
and orientation @f$ \theta @f$, the affine transformation @f$ (A,T)
@f$ reduces to the similarity transformation

@f[
     \mathbf{x} = m \sigma R(\theta) \hat{\mathbf{x}} + T
@f]

where @f$ R(\theta) @f$ is a counter-clockwise rotation of @f$ \theta
@f$ radians, @f$ m \mathcal{\sigma} @f$ is the size of a descriptor
bin in pixels, and @e m is the <b>descriptor magnification factor</b>
which expresses how much larger a descriptor bin is compared to
the scale of the keypoint @f$ \sigma @f$
(the default value is @e m = 3). Moreover, the
standard SIFT descriptor computes the image gradient at the scale of
the keypoints, which in the canonical frame is equivalent to a
smoothing of @f$ \hat \sigma = 1/m @f$. Finally, the default
Gaussian window size is set to have standard deviation
 @f$ \hat \sigma_\mathrm{win} = 2 @f$. This yields the formula

@f{eqnarray*}
 h(t,i,j)
 &=&
 m \sigma \int
 g_{\sigma_\mathrm{win}}(\mathbf{x} - T)\,
 w_\mathrm{ang}(\angle J(\mathbf{x}) - \theta  - \theta_t)\,
 w_{ij}\left(\frac{R(\theta)^\top \mathbf{x} - T}{m\sigma}\right)\,
 |J(\mathbf{x})|\,
 d\mathbf{x},
\\
\sigma_{\mathrm{win}} &=& m\sigma\hat \sigma_{\mathrm{win}},
\\
 J(\mathbf{x})
 &=& \nabla (g_{m \sigma \hat \sigma} * I)(\mathbf{x})
 = \nabla (g_{\sigma} * I)(\mathbf{x})
 = \nabla I_{\sigma} (\mathbf{x}).
@f}



**/

#include "sift.h"
#include "imopv.h"
#include "mathop.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

/** @internal @brief Use bilinear interpolation to compute orientations */
#define VL_SIFT_BILINEAR_ORIENTATIONS 1

#define EXPN_SZ  256          /**< ::fast_expn table size @internal */
#define EXPN_MAX 25.0         /**< ::fast_expn table max  @internal */
double expn_tab [EXPN_SZ+1] ; /**< ::fast_expn table      @internal */

#define NBO 8
#define NBP 4

#define log2(x) (log(x)/VL_LOG_OF_2)

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Fast @f$exp(-x)@f$ approximation
 **
 ** @param x argument.
 **
 ** The argument must be in the range [0, ::EXPN_MAX] .
 **
 ** @return approximation of @f$exp(-x)@f$.
 **/

VL_INLINE double
fast_expn (double x)
{
  double a,b,r ;
  int i ;
  /*assert(0 <= x && x <= EXPN_MAX) ;*/

  if (x > EXPN_MAX) return 0.0 ;

  x *= EXPN_SZ / EXPN_MAX ;
  i = (int)vl_floor_d (x) ;
  r = x - i ;
  a = expn_tab [i    ] ;
  b = expn_tab [i + 1] ;
  return a + r * (b - a) ;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Initialize tables for ::fast_expn
 **/

VL_INLINE void
fast_expn_init ()
{
  int k  ;
  for(k = 0 ; k < EXPN_SZ + 1 ; ++ k) {
    expn_tab [k] = exp (- (double) k * (EXPN_MAX / EXPN_SZ)) ;
  }
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy image, upsample rows and take transpose
 **
 ** @param dst     output image buffer.
 ** @param src     input image buffer.
 ** @param width   input image width.
 ** @param height  input image height.
 **
 ** The output image has dimensions @a height by 2 @a width (so the
 ** destination buffer must be at least as big as two times the
 ** input buffer).
 **
 ** Upsampling is performed by linear interpolation.
 **/

static void
copy_and_upsample_rows
(vl_sift_pix       *dst,
 vl_sift_pix const *src, int width, int height)
{
  int x, y ;
  vl_sift_pix a, b ;

  for(y = 0 ; y < height ; ++y) {
    b = a = *src++ ;
    for(x = 0 ; x < width - 1 ; ++x) {
      b = *src++ ;
      *dst = a ;             dst += height ;
      *dst = 0.5 * (a + b) ; dst += height ;
      a = b ;
    }
    *dst = b ; dst += height ;
    *dst = b ; dst += height ;
    dst += 1 - width * 2 * height ;
  }
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Smooth an image
 ** @param self        SIFT filter.
 ** @param outputImage output imgae buffer.
 ** @param tempImage   temporary image buffer.
 ** @param inputImage  input image buffer.
 ** @param width       input image width.
 ** @param height      input image height.
 ** @param sigma       smoothing.
 **/

static void
upsample_2x_direct
(vl_sift_pix       *dst,
 vl_sift_pix const *src, int width, int height)
{
  int x, y ;
  int dw = width * 2 ;
#ifdef __wasm_simd128__
  v128_t vhalf = wasm_f32x4_splat (0.5f) ;
#endif

  /* Fill even output rows directly from input rows */
  for (y = 0 ; y < height ; ++y) {
    vl_sift_pix const *srow = src + y * width ;
    vl_sift_pix *drow = dst + (2 * y) * dw ;
    x = 0 ;
#ifdef __wasm_simd128__
    for (; x + 4 < width ; x += 4) {
      v128_t a = wasm_v128_load (srow + x) ;
      v128_t b = wasm_v128_load (srow + x + 1) ;
      v128_t m = wasm_f32x4_mul (wasm_f32x4_add (a, b), vhalf) ;
      v128_t lo = wasm_i32x4_shuffle (a, m, 0, 4, 1, 5) ;
      v128_t hi = wasm_i32x4_shuffle (a, m, 2, 6, 3, 7) ;
      wasm_v128_store (drow + 2 * x, lo) ;
      wasm_v128_store (drow + 2 * x + 4, hi) ;
    }
#endif
    for (; x < width - 1 ; ++x) {
      vl_sift_pix a = srow[x] ;
      vl_sift_pix b = srow[x + 1] ;
      drow[2 * x]     = a ;
      drow[2 * x + 1] = 0.5f * (a + b) ;
    }
    drow[2 * width - 2] = srow[width - 1] ;
    drow[2 * width - 1] = srow[width - 1] ;
  }

  /* Fill odd output rows as average of adjacent even rows */
  for (y = 0 ; y < height - 1 ; ++y) {
    vl_sift_pix const *r0 = dst + (2 * y) * dw ;
    vl_sift_pix const *r1 = dst + (2 * y + 2) * dw ;
    vl_sift_pix *rm = dst + (2 * y + 1) * dw ;
    x = 0 ;
#ifdef __wasm_simd128__
    for (; x + 8 <= dw ; x += 8) {
      v128_t a0 = wasm_v128_load (r0 + x) ;
      v128_t b0 = wasm_v128_load (r1 + x) ;
      v128_t a1 = wasm_v128_load (r0 + x + 4) ;
      v128_t b1 = wasm_v128_load (r1 + x + 4) ;
      wasm_v128_store (rm + x,     wasm_f32x4_mul (wasm_f32x4_add (a0, b0), vhalf)) ;
      wasm_v128_store (rm + x + 4, wasm_f32x4_mul (wasm_f32x4_add (a1, b1), vhalf)) ;
    }
    for (; x + 4 <= dw ; x += 4) {
      wasm_v128_store (rm + x, wasm_f32x4_mul (wasm_f32x4_add (wasm_v128_load (r0 + x), wasm_v128_load (r1 + x)), vhalf)) ;
    }
#endif
    for (; x < dw ; ++x) {
      rm[x] = 0.5f * (r0[x] + r1[x]) ;
    }
  }
  memcpy (dst + (2 * height - 1) * dw, dst + (2 * height - 2) * dw, sizeof(vl_sift_pix) * dw) ;
}

static void
_vl_sift_smooth (VlSiftFilt * self,
                 vl_sift_pix * outputImage,
                 vl_sift_pix * tempImage,
                 vl_sift_pix const * inputImage,
                 vl_size width,
                 vl_size height,
                 double sigma)
{
  /* prepare Gaussian filter */
  if (self->gaussFilterSigma != sigma) {
    vl_uindex j ;
    vl_sift_pix acc = 0 ;
    if (self->gaussFilter) vl_free (self->gaussFilter) ;
    self->gaussFilterWidth = VL_MAX(ceil(4.0 * sigma), 1) ;
    self->gaussFilterSigma = sigma ;
    self->gaussFilter = vl_malloc (sizeof(vl_sift_pix) * (2 * self->gaussFilterWidth + 1)) ;

    for (j = 0 ; j < 2 * self->gaussFilterWidth + 1 ; ++j) {
      vl_sift_pix d = ((vl_sift_pix)((signed)j - (signed)self->gaussFilterWidth)) / ((vl_sift_pix)sigma) ;
      self->gaussFilter[j] = (vl_sift_pix) exp (- 0.5 * (d*d)) ;
      acc += self->gaussFilter[j] ;
    }
    for (j = 0 ; j < 2 * self->gaussFilterWidth + 1 ; ++j) {
      self->gaussFilter[j] /= acc ;
    }
  }

  if (self->gaussFilterWidth == 0) {
    memcpy (outputImage, inputImage, sizeof(vl_sift_pix) * width * height) ;
    return ;
  }

#ifdef __wasm_simd128__
  {
    int const W = (int)self->gaussFilterWidth ;
    int const w = (int)width ;
    int const h = (int)height ;
    vl_sift_pix const *f = self->gaussFilter + W ; /* symmetric: f[k] == f[-k] */
    static vl_sift_pix staticRowBuf[4352] ;
    vl_size need = (vl_size)(w + 2 * W + 8) ;
    vl_sift_pix *rowBuf = (need <= 4352) ? staticRowBuf : (vl_sift_pix*)vl_malloc (sizeof(vl_sift_pix) * need) ;
    vl_sift_pix *rp = rowBuf + W ;
    v128_t vc0 = wasm_f32x4_splat (f[0]) ;
    int y, x, k ;

    /* Pass 1: horizontal symmetric Gaussian blur (inputImage -> tempImage) */
    for (y = 0 ; y < h ; ++y) {
      vl_sift_pix const *srcRow = inputImage + (vl_size)y * width ;
      vl_sift_pix *dstRow = tempImage + (vl_size)y * width ;
      vl_sift_pix leftVal = srcRow[0] ;
      vl_sift_pix rightVal = srcRow[w - 1] ;
      for (k = 0 ; k < W ; ++k) rowBuf[k] = leftVal ;
      memcpy (rp, srcRow, sizeof(vl_sift_pix) * width) ;
      for (k = 0 ; k < W + 4 ; ++k) rp[w + k] = rightVal ;

      x = 0 ;
      for (; x + 8 <= w ; x += 8) {
        v128_t a0 = wasm_f32x4_mul (wasm_v128_load (rp + x), vc0) ;
        v128_t a1 = wasm_f32x4_mul (wasm_v128_load (rp + x + 4), vc0) ;
        for (k = 1 ; k <= W ; ++k) {
          v128_t ck = wasm_f32x4_splat (f[k]) ;
          v128_t s0 = wasm_f32x4_add (wasm_v128_load (rp + x - k), wasm_v128_load (rp + x + k)) ;
          v128_t s1 = wasm_f32x4_add (wasm_v128_load (rp + x + 4 - k), wasm_v128_load (rp + x + 4 + k)) ;
          a0 = wasm_f32x4_add (a0, wasm_f32x4_mul (s0, ck)) ;
          a1 = wasm_f32x4_add (a1, wasm_f32x4_mul (s1, ck)) ;
        }
        wasm_v128_store (dstRow + x, a0) ;
        wasm_v128_store (dstRow + x + 4, a1) ;
      }
      for (; x + 4 <= w ; x += 4) {
        v128_t a0 = wasm_f32x4_mul (wasm_v128_load (rp + x), vc0) ;
        for (k = 1 ; k <= W ; ++k) {
          v128_t ck = wasm_f32x4_splat (f[k]) ;
          v128_t s0 = wasm_f32x4_add (wasm_v128_load (rp + x - k), wasm_v128_load (rp + x + k)) ;
          a0 = wasm_f32x4_add (a0, wasm_f32x4_mul (s0, ck)) ;
        }
        wasm_v128_store (dstRow + x, a0) ;
      }
      for (; x < w ; ++x) {
        vl_sift_pix acc = rp[x] * f[0] ;
        for (k = 1 ; k <= W ; ++k) acc += (rp[x - k] + rp[x + k]) * f[k] ;
        dstRow[x] = acc ;
      }
    }

    if (rowBuf != staticRowBuf) vl_free (rowBuf) ;

    /* Pass 2: vertical symmetric Gaussian blur (tempImage -> outputImage),
     * strip-mined along x in 128-float tiles for L1 cache locality. */
    {
      int const TILE_X = 128 ;
      int x0 ;
      vl_sift_pix const *rm_stack[65] ;
      vl_sift_pix const *rp_stack[65] ;
      vl_sift_pix const **rm = (W <= 64) ? rm_stack : (vl_sift_pix const**)vl_malloc (sizeof(vl_sift_pix const*) * (W + 1)) ;
      vl_sift_pix const **rpp = (W <= 64) ? rp_stack : (vl_sift_pix const**)vl_malloc (sizeof(vl_sift_pix const*) * (W + 1)) ;

      for (x0 = 0 ; x0 < w ; x0 += TILE_X) {
        int x1 = (x0 + TILE_X < w) ? (x0 + TILE_X) : w ;
        for (y = 0 ; y < h ; ++y) {
          vl_sift_pix const *r0 = tempImage + (vl_size)y * width ;
          vl_sift_pix *outRow = outputImage + (vl_size)y * width ;

          if (y >= W && y < h - W) {
            x = x0 ;
            for (; x + 8 <= x1 ; x += 8) {
              v128_t a0 = wasm_f32x4_mul (wasm_v128_load (r0 + x), vc0) ;
              v128_t a1 = wasm_f32x4_mul (wasm_v128_load (r0 + x + 4), vc0) ;
              vl_sift_pix const *up = r0 + x - width ;
              vl_sift_pix const *dn = r0 + x + width ;
              for (k = 1 ; k <= W ; ++k) {
                v128_t ck = wasm_f32x4_splat (f[k]) ;
                v128_t s0 = wasm_f32x4_add (wasm_v128_load (up), wasm_v128_load (dn)) ;
                v128_t s1 = wasm_f32x4_add (wasm_v128_load (up + 4), wasm_v128_load (dn + 4)) ;
                a0 = wasm_f32x4_add (a0, wasm_f32x4_mul (s0, ck)) ;
                a1 = wasm_f32x4_add (a1, wasm_f32x4_mul (s1, ck)) ;
                up -= width ;
                dn += width ;
              }
              wasm_v128_store (outRow + x, a0) ;
              wasm_v128_store (outRow + x + 4, a1) ;
            }
            for (; x + 4 <= x1 ; x += 4) {
              v128_t a0 = wasm_f32x4_mul (wasm_v128_load (r0 + x), vc0) ;
              vl_sift_pix const *up = r0 + x - width ;
              vl_sift_pix const *dn = r0 + x + width ;
              for (k = 1 ; k <= W ; ++k) {
                v128_t ck = wasm_f32x4_splat (f[k]) ;
                a0 = wasm_f32x4_add (a0, wasm_f32x4_mul (wasm_f32x4_add (wasm_v128_load (up), wasm_v128_load (dn)), ck)) ;
                up -= width ;
                dn += width ;
              }
              wasm_v128_store (outRow + x, a0) ;
            }
            for (; x < x1 ; ++x) {
              vl_sift_pix acc = r0[x] * f[0] ;
              vl_sift_pix const *up = r0 + x - width ;
              vl_sift_pix const *dn = r0 + x + width ;
              for (k = 1 ; k <= W ; ++k) {
                acc += (*up + *dn) * f[k] ;
                up -= width ;
                dn += width ;
              }
              outRow[x] = acc ;
            }
          } else {
            for (k = 1 ; k <= W ; ++k) {
              int ym = y - k ; if (ym < 0) ym = 0 ;
              int yp = y + k ; if (yp >= h) yp = h - 1 ;
              rm[k] = tempImage + (vl_size)ym * width ;
              rpp[k] = tempImage + (vl_size)yp * width ;
            }
            x = x0 ;
            for (; x + 8 <= x1 ; x += 8) {
              v128_t a0 = wasm_f32x4_mul (wasm_v128_load (r0 + x), vc0) ;
              v128_t a1 = wasm_f32x4_mul (wasm_v128_load (r0 + x + 4), vc0) ;
              for (k = 1 ; k <= W ; ++k) {
                v128_t ck = wasm_f32x4_splat (f[k]) ;
                v128_t s0 = wasm_f32x4_add (wasm_v128_load (rm[k] + x), wasm_v128_load (rpp[k] + x)) ;
                v128_t s1 = wasm_f32x4_add (wasm_v128_load (rm[k] + x + 4), wasm_v128_load (rpp[k] + x + 4)) ;
                a0 = wasm_f32x4_add (a0, wasm_f32x4_mul (s0, ck)) ;
                a1 = wasm_f32x4_add (a1, wasm_f32x4_mul (s1, ck)) ;
              }
              wasm_v128_store (outRow + x, a0) ;
              wasm_v128_store (outRow + x + 4, a1) ;
            }
            for (; x + 4 <= x1 ; x += 4) {
              v128_t a0 = wasm_f32x4_mul (wasm_v128_load (r0 + x), vc0) ;
              for (k = 1 ; k <= W ; ++k) {
                v128_t ck = wasm_f32x4_splat (f[k]) ;
                a0 = wasm_f32x4_add (a0, wasm_f32x4_mul (wasm_f32x4_add (wasm_v128_load (rm[k] + x), wasm_v128_load (rpp[k] + x)), ck)) ;
              }
              wasm_v128_store (outRow + x, a0) ;
            }
            for (; x < x1 ; ++x) {
              vl_sift_pix acc = r0[x] * f[0] ;
              for (k = 1 ; k <= W ; ++k) acc += (rm[k][x] + rpp[k][x]) * f[k] ;
              outRow[x] = acc ;
            }
          }
        }
      }
      if (rm != rm_stack) { vl_free ((void*)rm) ; vl_free ((void*)rpp) ; }
    }
    return ;
  }
#endif

  vl_imconvcol_vf (tempImage, height,
                   inputImage, width, height, width,
                   self->gaussFilter,
                   - self->gaussFilterWidth, self->gaussFilterWidth,
                   1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;

  vl_imconvcol_vf (outputImage, width,
                   tempImage, height, width, height,
                   self->gaussFilter,
                   - self->gaussFilterWidth, self->gaussFilterWidth,
                   1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy and downsample an image
 **
 ** @param dst    output imgae buffer.
 ** @param src    input  image buffer.
 ** @param width  input  image width.
 ** @param height input  image height.
 ** @param d      octaves (non negative).
 **
 ** The function downsamples the image @a d times, reducing it to @c
 ** 1/2^d of its original size. The parameters @a width and @a height
 ** are the size of the input image. The destination image @a dst is
 ** assumed to be <code>floor(width/2^d)</code> pixels wide and
 ** <code>floor(height/2^d)</code> pixels high.
 **/

static void
copy_and_downsample
(vl_sift_pix       *dst,
 vl_sift_pix const *src,
 int width, int height, int d)
{
  int x, y ;

  d = 1 << d ; /* d = 2^d */
  for(y = 0 ; y < height ; y+=d) {
    vl_sift_pix const * srcrowp = src + y * width ;
    for(x = 0 ; x < width - (d-1) ; x+=d) {
      *dst++ = *srcrowp ;
      srcrowp += d ;
    }
  }
}

/** ------------------------------------------------------------------
 ** @brief Create a new SIFT filter
 **
 ** @param width    image width.
 ** @param height   image height.
 ** @param noctaves number of octaves.
 ** @param nlevels  number of levels per octave.
 ** @param o_min    first octave index.
 **
 ** The function allocates and returns a new SIFT filter for the
 ** specified image and scale space geometry.
 **
 ** Setting @a O to a negative value sets the number of octaves to the
 ** maximum possible value depending on the size of the image.
 **
 ** @return the new SIFT filter.
 ** @sa ::vl_sift_delete().
 **/

VL_EXPORT
VlSiftFilt *
vl_sift_new (int width, int height,
             int noctaves, int nlevels,
             int o_min)
{
  VlSiftFilt *f = vl_malloc (sizeof(VlSiftFilt)) ;

  int w   = VL_SHIFT_LEFT (width,  -o_min) ;
  int h   = VL_SHIFT_LEFT (height, -o_min) ;
  int nel = w * h ;

  /* negative value O => calculate max. value */
  if (noctaves < 0) {
    noctaves = VL_MAX (floor (log2 (VL_MIN(width, height))) - o_min - 3, 1) ;
  }

  f-> width   = width ;
  f-> height  = height ;
  f-> O       = noctaves ;
  f-> S       = nlevels ;
  f-> o_min   = o_min ;
  f-> s_min   = -1 ;
  f-> s_max   = nlevels + 1 ;
  f-> o_cur   = o_min ;

  f-> temp    = vl_malloc (sizeof(vl_sift_pix) * nel    ) ;
  f-> octave  = vl_malloc (sizeof(vl_sift_pix) * nel
                        * (f->s_max - f->s_min + 1)  ) ;
  f-> dog     = vl_malloc (sizeof(vl_sift_pix) * nel
                        * (f->s_max - f->s_min    )  ) ;
  f-> grad    = vl_malloc (sizeof(vl_sift_pix) * nel * 2
                        * (f->s_max - f->s_min    )  ) ;

  f-> sigman  = 0.5 ;
  f-> sigmak  = pow (2.0, 1.0 / nlevels) ;
  f-> sigma0  = 1.6 * f->sigmak ;
  f-> dsigma0 = f->sigma0 * sqrt (1.0 - 1.0 / (f->sigmak*f->sigmak)) ;

  f-> gaussFilter = NULL ;
  f-> gaussFilterSigma = 0 ;
  f-> gaussFilterWidth = 0 ;

  f-> octave_width  = 0 ;
  f-> octave_height = 0 ;

  f-> keys     = 0 ;
  f-> nkeys    = 0 ;
  f-> keys_res = 0 ;

  f-> peak_thresh = 0.0 ;
  f-> edge_thresh = 10.0 ;
  f-> norm_thresh = 0.0 ;
  f-> magnif      = 3.0 ;
  f-> windowSize  = NBP / 2 ;

  f-> grad_o  = o_min - 1 ;

  /* initialize fast_expn stuff */
  fast_expn_init () ;

  return f ;
}

/** -------------------------------------------------------------------
 ** @brief Delete SIFT filter
 **
 ** @param f SIFT filter to delete.
 **
 ** The function frees the resources allocated by ::vl_sift_new().
 **/

VL_EXPORT
void
vl_sift_delete (VlSiftFilt* f)
{
  if (f) {
    if (f->keys) vl_free (f->keys) ;
    if (f->grad) vl_free (f->grad) ;
    if (f->dog) vl_free (f->dog) ;
    if (f->octave) vl_free (f->octave) ;
    if (f->temp) vl_free (f->temp) ;
    if (f->gaussFilter) vl_free (f->gaussFilter) ;
    vl_free (f) ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Start processing a new image
 **
 ** @param f  SIFT filter.
 ** @param im image data.
 **
 ** The function starts processing a new image by computing its
 ** Gaussian scale space at the lower octave. It also empties the
 ** internal keypoint buffer.
 **
 ** @return error code. The function returns ::VL_ERR_EOF if there are
 ** no more octaves to process.
 **
 ** @sa ::vl_sift_process_next_octave().
 **/

VL_EXPORT
int
vl_sift_process_first_octave (VlSiftFilt *f, vl_sift_pix const *im)
{
  int o, s, h, w ;
  double sa, sb ;
  vl_sift_pix *octave ;

  /* shortcuts */
  vl_sift_pix *temp   = f-> temp ;
  int width           = f-> width ;
  int height          = f-> height ;
  int o_min           = f-> o_min ;
  int s_min           = f-> s_min ;
  int s_max           = f-> s_max ;
  double sigma0       = f-> sigma0 ;
  double sigmak       = f-> sigmak ;
  double sigman       = f-> sigman ;
  double dsigma0      = f-> dsigma0 ;

  /* restart from the first */
  f->o_cur = o_min ;
  f->nkeys = 0 ;
  w = f-> octave_width  = VL_SHIFT_LEFT(f->width,  - f->o_cur) ;
  h = f-> octave_height = VL_SHIFT_LEFT(f->height, - f->o_cur) ;

  /* is there at least one octave? */
  if (f->O == 0)
    return VL_ERR_EOF ;

  /* ------------------------------------------------------------------
   *                     Compute the first sublevel of the first octave
   * --------------------------------------------------------------- */

  /*
   * If the first octave has negative index, we upscale the image; if
   * the first octave has positive index, we downscale the image; if
   * the first octave has index zero, we just copy the image.
   */

  octave = vl_sift_get_octave (f, s_min) ;

  if (o_min < 0) {
    /* double once */
    upsample_2x_direct (octave, im, width, height) ;

    /* double more */
    for(o = -1 ; o > o_min ; --o) {
      copy_and_upsample_rows (temp, octave,
                              width << -o,      height << -o ) ;
      copy_and_upsample_rows (octave, temp,
                              width << -o, 2 * (height << -o)) ;
    }
  }
  else if (o_min > 0) {
    /* downsample */
    copy_and_downsample (octave, im, width, height, o_min) ;
  }
  else {
    /* direct copy */
    memcpy(octave, im, sizeof(vl_sift_pix) * width * height) ;
  }

  /*
   * Here we adjust the smoothing of the first level of the octave.
   * The input image is assumed to have nominal smoothing equal to
   * f->simgan.
   */

  sa = sigma0 * pow (sigmak,   s_min) ;
  sb = sigman * pow (2.0,    - o_min) ;

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb) ;
    _vl_sift_smooth (f, octave, temp, octave, w, h, sd) ;
  }

  /* -----------------------------------------------------------------
   *                                          Compute the first octave
   * -------------------------------------------------------------- */

  for(s = s_min + 1 ; s <= s_max ; ++s) {
    double sd = dsigma0 * pow (sigmak, s) ;
    _vl_sift_smooth (f, vl_sift_get_octave(f, s), temp,
                     vl_sift_get_octave(f, s - 1), w, h, sd) ;
  }

  return VL_ERR_OK ;
}

/** ------------------------------------------------------------------
 ** @brief Process next octave
 **
 ** @param f SIFT filter.
 **
 ** The function computes the next octave of the Gaussian scale space.
 ** Notice that this clears the record of any feature detected in the
 ** previous octave.
 **
 ** @return error code. The function returns the error
 ** ::VL_ERR_EOF when there are no more octaves to process.
 **
 ** @sa ::vl_sift_process_first_octave().
 **/

VL_EXPORT
int
vl_sift_process_next_octave (VlSiftFilt *f)
{

  int s, h, w, s_best ;
  double sa, sb ;
  vl_sift_pix *octave, *pt ;

  /* shortcuts */
  vl_sift_pix *temp   = f-> temp ;
  int O               = f-> O ;
  int S               = f-> S ;
  int o_min           = f-> o_min ;
  int s_min           = f-> s_min ;
  int s_max           = f-> s_max ;
  double sigma0       = f-> sigma0 ;
  double sigmak       = f-> sigmak ;
  double dsigma0      = f-> dsigma0 ;

  /* is there another octave ? */
  if (f->o_cur == o_min + O - 1)
    return VL_ERR_EOF ;

  /* retrieve base */
  s_best = VL_MIN(s_min + S, s_max) ;
  w      = vl_sift_get_octave_width  (f) ;
  h      = vl_sift_get_octave_height (f) ;
  pt     = vl_sift_get_octave        (f, s_best) ;
  octave = vl_sift_get_octave        (f, s_min) ;

  /* next octave */
  copy_and_downsample (octave, pt, w, h, 1) ;

  f-> o_cur            += 1 ;
  f-> nkeys             = 0 ;
  w = f-> octave_width  = VL_SHIFT_LEFT(f->width,  - f->o_cur) ;
  h = f-> octave_height = VL_SHIFT_LEFT(f->height, - f->o_cur) ;

  sa = sigma0 * powf (sigmak, s_min     ) ;
  sb = sigma0 * powf (sigmak, s_best - S) ;

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb) ;
    _vl_sift_smooth (f, octave, temp, octave, w, h, sd) ;
  }

  /* ------------------------------------------------------------------
   *                                                        Fill octave
   * --------------------------------------------------------------- */

  for(s = s_min + 1 ; s <= s_max ; ++s) {
    double sd = dsigma0 * pow (sigmak, s) ;
    _vl_sift_smooth (f, vl_sift_get_octave(f, s), temp,
                     vl_sift_get_octave(f, s - 1), w, h, sd) ;
  }

  return VL_ERR_OK ;
}

/** ------------------------------------------------------------------
 ** @brief Detect keypoints
 **
 ** The function detect keypoints in the current octave filling the
 ** internal keypoint buffer. Keypoints can be retrieved by
 ** ::vl_sift_get_keypoints().
 **
 ** @param f SIFT filter.
 **/

VL_EXPORT
void
vl_sift_detect (VlSiftFilt * f)
{
  vl_sift_pix* dog   = f-> dog ;
  int          s_min = f-> s_min ;
  int          s_max = f-> s_max ;
  int          w     = f-> octave_width ;
  int          h     = f-> octave_height ;
  double       te    = f-> edge_thresh ;
  double       tp    = f-> peak_thresh ;

  int const    xo    = 1 ;      /* x-stride */
  int const    yo    = w ;      /* y-stride */
  int const    so    = w * h ;  /* s-stride */

  double       xper  = pow (2.0, f->o_cur) ;

  int x, y, s, i, ii, jj ;
  vl_sift_pix *pt, v ;
  VlSiftKeypoint *k ;

  /* clear current list */
  f-> nkeys = 0 ;

  /* compute difference of gaussian (DoG) */
  pt = f-> dog ;
  for (s = s_min ; s <= s_max - 1 ; ++s) {
    vl_sift_pix* src_a = vl_sift_get_octave (f, s    ) ;
    vl_sift_pix* src_b = vl_sift_get_octave (f, s + 1) ;
    int count = w * h ;
    int idx = 0 ;
#ifdef __wasm_simd128__
    for (; idx + 8 <= count ; idx += 8) {
      wasm_v128_store (pt + idx,     wasm_f32x4_sub (wasm_v128_load (src_b + idx),     wasm_v128_load (src_a + idx))) ;
      wasm_v128_store (pt + idx + 4, wasm_f32x4_sub (wasm_v128_load (src_b + idx + 4), wasm_v128_load (src_a + idx + 4))) ;
    }
    for (; idx + 4 <= count ; idx += 4) {
      wasm_v128_store (pt + idx, wasm_f32x4_sub (wasm_v128_load (src_b + idx), wasm_v128_load (src_a + idx))) ;
    }
#endif
    for (; idx < count ; ++idx) {
      pt[idx] = src_b[idx] - src_a[idx] ;
    }
    pt += count ;
  }

  /* -----------------------------------------------------------------
   *                                          Find local maxima of DoG
   * -------------------------------------------------------------- */

  /* start from dog [1,1,s_min+1] */
  pt  = dog + xo + yo + so ;
  {
#ifdef __wasm_simd128__
    float const thresh_lo = (float)(0.8 * tp) - 1e-6f ;
    v128_t const v_pos = wasm_f32x4_splat (thresh_lo) ;
    v128_t const v_neg = wasm_f32x4_splat (-thresh_lo) ;
#endif

  for(s = s_min + 1 ; s <= s_max - 2 ; ++s) {
    for(y = 1 ; y < h - 1 ; ++y) {
      x = 1 ;
#define CHECK_NEIGHBORS_AT(P,CMP,SGN)               \
        ( v CMP ## = SGN 0.8 * tp &&                \
          v CMP *((P) + xo) &&                      \
          v CMP *((P) - xo) &&                      \
          v CMP *((P) + so) &&                      \
          v CMP *((P) - so) &&                      \
          v CMP *((P) + yo) &&                      \
          v CMP *((P) - yo) &&                      \
                                                    \
          v CMP *((P) + yo + xo) &&                 \
          v CMP *((P) + yo - xo) &&                 \
          v CMP *((P) - yo + xo) &&                 \
          v CMP *((P) - yo - xo) &&                 \
                                                    \
          v CMP *((P) + xo      + so) &&            \
          v CMP *((P) - xo      + so) &&            \
          v CMP *((P) + yo      + so) &&            \
          v CMP *((P) - yo      + so) &&            \
          v CMP *((P) + yo + xo + so) &&            \
          v CMP *((P) + yo - xo + so) &&            \
          v CMP *((P) - yo + xo + so) &&            \
          v CMP *((P) - yo - xo + so) &&            \
                                                    \
          v CMP *((P) + xo      - so) &&            \
          v CMP *((P) - xo      - so) &&            \
          v CMP *((P) + yo      - so) &&            \
          v CMP *((P) - yo      - so) &&            \
          v CMP *((P) + yo + xo - so) &&            \
          v CMP *((P) + yo - xo - so) &&            \
          v CMP *((P) - yo + xo - so) &&            \
          v CMP *((P) - yo - xo - so) )

#ifdef __wasm_simd128__
      for(; x + 4 <= w - 1 ; x += 4) {
        v128_t vv = wasm_v128_load (pt) ;
        v128_t vl = wasm_v128_load (pt - 1) ;
        v128_t vr = wasm_v128_load (pt + 1) ;
        v128_t is_max = wasm_v128_and (wasm_f32x4_ge (vv, v_pos),
                          wasm_v128_and (wasm_f32x4_gt (vv, vl), wasm_f32x4_gt (vv, vr))) ;
        v128_t is_min = wasm_v128_and (wasm_f32x4_le (vv, v_neg),
                          wasm_v128_and (wasm_f32x4_lt (vv, vl), wasm_f32x4_lt (vv, vr))) ;
        int mask = wasm_i32x4_bitmask (wasm_v128_or (is_max, is_min)) ;
        if (__builtin_expect (mask != 0, 0)) {
          int lane ;
          for (lane = 0 ; lane < 4 ; ++lane) {
            if (mask & (1 << lane)) {
              vl_sift_pix *p = pt + lane ;
              v = *p ;
              if (CHECK_NEIGHBORS_AT(p,>,+) || CHECK_NEIGHBORS_AT(p,<,-)) {
                if (f->nkeys >= f->keys_res) {
                  f->keys_res += 500 ;
                  if (f->keys) {
                    f->keys = vl_realloc (f->keys, f->keys_res * sizeof(VlSiftKeypoint)) ;
                  } else {
                    f->keys = vl_malloc (f->keys_res * sizeof(VlSiftKeypoint)) ;
                  }
                }
                k = f->keys + (f->nkeys ++) ;
                k-> ix = x + lane ;
                k-> iy = y ;
                k-> is = s ;
              }
            }
          }
        }
        pt += 4 ;
      }
#endif
      for(; x < w - 1 ; ++x) {
        v = *pt ;
        if (CHECK_NEIGHBORS_AT(pt,>,+) ||
            CHECK_NEIGHBORS_AT(pt,<,-) ) {

          /* make room for more keypoints */
          if (f->nkeys >= f->keys_res) {
            f->keys_res += 500 ;
            if (f->keys) {
              f->keys = vl_realloc (f->keys,
                                    f->keys_res *
                                    sizeof(VlSiftKeypoint)) ;
            } else {
              f->keys = vl_malloc (f->keys_res *
                                   sizeof(VlSiftKeypoint)) ;
            }
          }

          k = f->keys + (f->nkeys ++) ;

          k-> ix = x ;
          k-> iy = y ;
          k-> is = s ;
        }
        pt += 1 ;
      }
      pt += 2 ;
    }
    pt += 2 * yo ;
  }
  }

  /* -----------------------------------------------------------------
   *                                               Refine local maxima
   * -------------------------------------------------------------- */

  /* this pointer is used to write the keypoints back */
  k = f->keys ;

  for (i = 0 ; i < f->nkeys ; ++i) {

    int x = f-> keys [i] .ix ;
    int y = f-> keys [i] .iy ;
    int s = f-> keys [i]. is ;

    double Dx=0,Dy=0,Ds=0,Dxx=0,Dyy=0,Dss=0,Dxy=0,Dxs=0,Dys=0 ;
    double A [3*3], b [3] ;

    int dx = 0 ;
    int dy = 0 ;

    int iter, i, j ;

    for (iter = 0 ; iter < 5 ; ++iter) {

      x += dx ;
      y += dy ;

      pt = dog
        + xo * x
        + yo * y
        + so * (s - s_min) ;

      /** @brief Index GSS @internal */
#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))

      /** @brief Index matrix A @internal */
#define Aat(i,j)     (A[(i)+(j)*3])

      /* compute the gradient */
      Dx = 0.5 * (at(+1,0,0) - at(-1,0,0)) ;
      Dy = 0.5 * (at(0,+1,0) - at(0,-1,0));
      Ds = 0.5 * (at(0,0,+1) - at(0,0,-1)) ;

      /* compute the Hessian */
      Dxx = (at(+1,0,0) + at(-1,0,0) - 2.0 * at(0,0,0)) ;
      Dyy = (at(0,+1,0) + at(0,-1,0) - 2.0 * at(0,0,0)) ;
      Dss = (at(0,0,+1) + at(0,0,-1) - 2.0 * at(0,0,0)) ;

      Dxy = 0.25 * ( at(+1,+1,0) + at(-1,-1,0) - at(-1,+1,0) - at(+1,-1,0) ) ;
      Dxs = 0.25 * ( at(+1,0,+1) + at(-1,0,-1) - at(-1,0,+1) - at(+1,0,-1) ) ;
      Dys = 0.25 * ( at(0,+1,+1) + at(0,-1,-1) - at(0,-1,+1) - at(0,+1,-1) ) ;

      /* solve linear system ....................................... */
      Aat(0,0) = Dxx ;
      Aat(1,1) = Dyy ;
      Aat(2,2) = Dss ;
      Aat(0,1) = Aat(1,0) = Dxy ;
      Aat(0,2) = Aat(2,0) = Dxs ;
      Aat(1,2) = Aat(2,1) = Dys ;

      b[0] = - Dx ;
      b[1] = - Dy ;
      b[2] = - Ds ;

      /* Gauss elimination */
      for(j = 0 ; j < 3 ; ++j) {
        double maxa    = 0 ;
        double maxabsa = 0 ;
        int    maxi    = -1 ;
        double tmp ;

        /* look for the maximally stable pivot */
        for (i = j ; i < 3 ; ++i) {
          double a    = Aat (i,j) ;
          double absa = vl_abs_d (a) ;
          if (absa > maxabsa) {
            maxa    = a ;
            maxabsa = absa ;
            maxi    = i ;
          }
        }

        /* if singular give up */
        if (maxabsa < 1e-10f) {
          b[0] = 0 ;
          b[1] = 0 ;
          b[2] = 0 ;
          break ;
        }

        i = maxi ;

        /* swap j-th row with i-th row and normalize j-th row */
        for(jj = j ; jj < 3 ; ++jj) {
          tmp = Aat(i,jj) ; Aat(i,jj) = Aat(j,jj) ; Aat(j,jj) = tmp ;
          Aat(j,jj) /= maxa ;
        }
        tmp = b[j] ; b[j] = b[i] ; b[i] = tmp ;
        b[j] /= maxa ;

        /* elimination */
        for (ii = j+1 ; ii < 3 ; ++ii) {
          double x = Aat(ii,j) ;
          for (jj = j ; jj < 3 ; ++jj) {
            Aat(ii,jj) -= x * Aat(j,jj) ;
          }
          b[ii] -= x * b[j] ;
        }
      }

      /* backward substitution */
      for (i = 2 ; i > 0 ; --i) {
        double x = b[i] ;
        for (ii = i-1 ; ii >= 0 ; --ii) {
          b[ii] -= x * Aat(ii,i) ;
        }
      }

      /* .......................................................... */
      /* If the translation of the keypoint is big, move the keypoint
       * and re-iterate the computation. Otherwise we are all set.
       */

      dx= ((b[0] >  0.6 && x < w - 2) ?  1 : 0)
        + ((b[0] < -0.6 && x > 1    ) ? -1 : 0) ;

      dy= ((b[1] >  0.6 && y < h - 2) ?  1 : 0)
        + ((b[1] < -0.6 && y > 1    ) ? -1 : 0) ;

      if (dx == 0 && dy == 0) break ;
    }

    /* check threshold and other conditions */
    {
      double val   = at(0,0,0)
        + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]) ;
      double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy) ;
      double xn = x + b[0] ;
      double yn = y + b[1] ;
      double sn = s + b[2] ;

      vl_bool good =
        vl_abs_d (val)  > tp                  &&
        score           < (te+1)*(te+1)/te    &&
        score           >= 0                  &&
        vl_abs_d (b[0]) <  1.5                &&
        vl_abs_d (b[1]) <  1.5                &&
        vl_abs_d (b[2]) <  1.5                &&
        xn              >= 0                  &&
        xn              <= w - 1              &&
        yn              >= 0                  &&
        yn              <= h - 1              &&
        sn              >= s_min              &&
        sn              <= s_max ;

      if (good) {
        k-> o     = f->o_cur ;
        k-> ix    = x ;
        k-> iy    = y ;
        k-> is    = s ;
        k-> s     = sn ;
        k-> x     = xn * xper ;
        k-> y     = yn * xper ;
        k-> sigma = f->sigma0 * pow (2.0, sn/f->S) * xper ;
        ++ k ;
      }

    } /* done checking */
  } /* next keypoint to refine */

  /* update keypoint count */
  f-> nkeys = (int)(k - f->keys) ;
}


/** ------------------------------------------------------------------
 ** @internal
 ** @brief Update gradients to current GSS octave
 **
 ** @param f SIFT filter.
 **
 ** The function makes sure that the gradient buffer is up-to-date
 ** with the current GSS data.
 **
 ** @remark The minimum octave size is 2x2xS.
 **/

static void
update_gradient (VlSiftFilt *f)
{
  int       s_min = f->s_min ;
  int       s_max = f->s_max ;
  int       w     = vl_sift_get_octave_width  (f) ;
  int       h     = vl_sift_get_octave_height (f) ;
  int const xo    = 1 ;
  int const yo    = w ;
  int const so    = h * w ;
  int y, s ;

  if (f->grad_o == f->o_cur) return ;

  for (s  = s_min + 1 ;
       s <= s_max - 2 ; ++ s) {

    vl_sift_pix *src, *end, *grad, gx, gy ;

#define SAVE_BACK                                                       \
    *grad++ = vl_fast_sqrt_f (gx*gx + gy*gy) ;                          \
    *grad++ = vl_mod_2pi_f   (vl_fast_atan2_f (gy, gx) + 2*VL_PI) ;     \
    ++src ;                                                             \

    src  = vl_sift_get_octave (f,s) ;
    grad = f->grad + 2 * so * (s - s_min -1) ;

    /* first pixel of the first row */
    gx = src[+xo] - src[0] ;
    gy = src[+yo] - src[0] ;
    SAVE_BACK ;

    /* middle pixels of the  first row */
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy =        src[+yo] - src[0] ;
      SAVE_BACK ;
    }

    /* last pixel of the first row */
    gx = src[0]   - src[-xo] ;
    gy = src[+yo] - src[0] ;
    SAVE_BACK ;

    for (y = 1 ; y < h -1 ; ++y) {

      /* first pixel of the middle rows */
      gx =        src[+xo] - src[0] ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;

      /* middle pixels of the middle rows */
      end = (src - 1) + w - 1 ;
#ifdef __wasm_simd128__
      {
        v128_t const vhalf = wasm_f32x4_splat (0.5f) ;
        v128_t const vzero = wasm_f32x4_splat (0.0f) ;
        v128_t const veps  = wasm_f32x4_splat (VL_EPSILON_F) ;
        v128_t const vc3   = wasm_f32x4_splat (0.1821f) ;
        v128_t const vc1   = wasm_f32x4_splat (0.9675f) ;
        v128_t const vpi4  = wasm_f32x4_splat ((float)(VL_PI / 4.0)) ;
        v128_t const v3pi4 = wasm_f32x4_splat ((float)(3.0 * VL_PI / 4.0)) ;
        v128_t const v2pi  = wasm_f32x4_splat ((float)(2.0 * VL_PI)) ;
        while (src + 4 <= end) {
          v128_t vgx = wasm_f32x4_mul (vhalf, wasm_f32x4_sub (wasm_v128_load (src + xo), wasm_v128_load (src - xo))) ;
          v128_t vgy = wasm_f32x4_mul (vhalf, wasm_f32x4_sub (wasm_v128_load (src + yo), wasm_v128_load (src - yo))) ;
          v128_t mag = wasm_f32x4_sqrt (wasm_f32x4_add (wasm_f32x4_mul (vgx, vgx), wasm_f32x4_mul (vgy, vgy))) ;
          v128_t abs_y = wasm_f32x4_add (wasm_f32x4_abs (vgy), veps) ;
          v128_t ge0 = wasm_f32x4_ge (vgx, vzero) ;
          v128_t num = wasm_v128_bitselect (wasm_f32x4_sub (vgx, abs_y), wasm_f32x4_add (vgx, abs_y), ge0) ;
          v128_t den = wasm_v128_bitselect (wasm_f32x4_add (vgx, abs_y), wasm_f32x4_sub (abs_y, vgx), ge0) ;
          v128_t r   = wasm_f32x4_div (num, den) ;
          v128_t ang = wasm_f32x4_add (wasm_v128_bitselect (vpi4, v3pi4, ge0),
                         wasm_f32x4_mul (wasm_f32x4_sub (wasm_f32x4_mul (vc3, wasm_f32x4_mul (r, r)), vc1), r)) ;
          ang = wasm_v128_bitselect (wasm_f32x4_neg (ang), ang, wasm_f32x4_lt (vgy, vzero)) ;
          ang = wasm_v128_bitselect (wasm_f32x4_add (ang, v2pi), ang, wasm_f32x4_lt (ang, vzero)) ;
          ang = wasm_v128_bitselect (wasm_f32x4_sub (ang, v2pi), ang, wasm_f32x4_ge (ang, v2pi)) ;
          wasm_v128_store (grad,     wasm_i32x4_shuffle (mag, ang, 0, 4, 1, 5)) ;
          wasm_v128_store (grad + 4, wasm_i32x4_shuffle (mag, ang, 2, 6, 3, 7)) ;
          grad += 8 ;
          src  += 4 ;
        }
      }
#endif
      while (src < end) {
        gx = 0.5 * (src[+xo] - src[-xo]) ;
        gy = 0.5 * (src[+yo] - src[-yo]) ;
        SAVE_BACK ;
      }

      /* last pixel of the middle row */
      gx =        src[0]   - src[-xo] ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;
    }

    /* first pixel of the last row */
    gx = src[+xo] - src[0] ;
    gy = src[  0] - src[-yo] ;
    SAVE_BACK ;

    /* middle pixels of the last row */
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy =        src[0]   - src[-yo] ;
      SAVE_BACK ;
    }

    /* last pixel of the last row */
    gx = src[0]   - src[-xo] ;
    gy = src[0]   - src[-yo] ;
    SAVE_BACK ;
  }
  f->grad_o = f->o_cur ;
}

/** ------------------------------------------------------------------
 ** @brief Calculate the keypoint orientation(s)
 **
 ** @param f        SIFT filter.
 ** @param angles   orientations (output).
 ** @param k        keypoint.
 **
 ** The function computes the orientation(s) of the keypoint @a k.
 ** The function returns the number of orientations found (up to
 ** four). The orientations themselves are written to the vector @a
 ** angles.
 **
 ** @remark The function requires the keypoint octave @a k->o to be
 ** equal to the filter current octave ::vl_sift_get_octave. If this
 ** is not the case, the function returns zero orientations.
 **
 ** @remark The function requires the keypoint scale level @c k->s to
 ** be in the range @c s_min+1 and @c s_max-2 (where usually @c
 ** s_min=0 and @c s_max=S+2). If this is not the case, the function
 ** returns zero orientations.
 **
 ** @return number of orientations found.
 **/

VL_EXPORT
int
vl_sift_calc_keypoint_orientations (VlSiftFilt *f,
                                    double angles [4],
                                    VlSiftKeypoint const *k)
{
  double const winf   = 1.5 ;
  double       xper   = pow (2.0, f->o_cur) ;

  int          w      = f-> octave_width ;
  int          h      = f-> octave_height ;
  int const    xo     = 2 ;         /* x-stride */
  int const    yo     = 2 * w ;     /* y-stride */
  int const    so     = 2 * w * h ; /* s-stride */
  double       x      = k-> x     / xper ;
  double       y      = k-> y     / xper ;
  double       sigma  = k-> sigma / xper ;

  int          xi     = (int) (x + 0.5) ;
  int          yi     = (int) (y + 0.5) ;
  int          si     = k-> is ;

  double const sigmaw = winf * sigma ;
  int          W      = VL_MAX(floor (3.0 * sigmaw), 1) ;

  int          nangles= 0 ;

  enum {nbins = 36} ;

  double hist [nbins], maxh ;
  vl_sift_pix const * pt ;
  int xs, ys, iter, i ;

  /* skip if the keypoint octave is not current */
  if(k->o != f->o_cur)
    return 0 ;

  /* skip the keypoint if it is out of bounds */
  if(xi < 0            ||
     xi > w - 1        ||
     yi < 0            ||
     yi > h - 1        ||
     si < f->s_min + 1 ||
     si > f->s_max - 2  ) {
    return 0 ;
  }

  /* make gradient up to date */
  update_gradient (f) ;

  /* clear histogram */
  memset (hist, 0, sizeof(double) * nbins) ;

  /* compute orientation histogram */
  pt = f-> grad + xo*xi + yo*yi + so*(si - f->s_min - 1) ;

#undef  at
#define at(dx,dy) (*(pt + xo * (dx) + yo * (dy)))

  {
    double const inv_2sigmaw2 = 1.0 / (2.0 * sigmaw * sigmaw) ;
    double const nbins_over_2pi = nbins / (2.0 * VL_PI) ;
    double const max_r2 = (double)W * W + 0.6 ;
    int const ys_min = VL_MAX (- W,       - yi) ;
    int const ys_max = VL_MIN (+ W, h - 1 - yi) ;
    int const xs_min = VL_MAX (- W,       - xi) ;
    int const xs_max = VL_MIN (+ W, w - 1 - xi) ;

    for (ys = ys_min ; ys <= ys_max ; ++ys) {
      double dy = (double)(yi + ys) - y ;
      double dy2 = dy * dy ;
      vl_sift_pix const *row = pt + ys * yo + xs_min * xo ;
      double dx = (double)(xi + xs_min) - x ;

      for (xs = xs_min ; xs <= xs_max ; ++xs, dx += 1.0, row += xo) {
        double r2 = dx * dx + dy2 ;
        double wgt, mod, ang, fbin ;

        /* limit to a circular window */
        if (r2 >= max_r2) continue ;

        wgt  = fast_expn (r2 * inv_2sigmaw2) ;
        mod  = row[0] ;
        ang  = row[1] ;
        fbin = ang * nbins_over_2pi ;

#if defined(VL_SIFT_BILINEAR_ORIENTATIONS)
        {
          int bin = (int) vl_floor_d (fbin - 0.5) ;
          double rbin = fbin - bin - 0.5 ;
          double mw = mod * wgt ;
          int b0 = bin ; if (b0 < 0) b0 += nbins ; else if (b0 >= nbins) b0 -= nbins ;
          int b1 = b0 + 1 ; if (b1 >= nbins) b1 -= nbins ;
          hist [b0] += (1.0 - rbin) * mw ;
          hist [b1] += (      rbin) * mw ;
        }
#else
        {
          int bin = (int) vl_floor_d (fbin) ;
          hist [(bin + nbins) % nbins] += mod * wgt ;
        }
#endif
      } /* for xs */
    } /* for ys */
  }

  /* smooth histogram */
  for (iter = 0; iter < 6; iter ++) {
    double prev  = hist [nbins - 1] ;
    double first = hist [0] ;
    int i ;
    for (i = 0; i < nbins - 1; i++) {
      double newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
      prev = hist[i] ;
      hist[i] = newh ;
    }
    hist[i] = (prev + hist[i] + first) / 3.0 ;
  }

  /* find the histogram maximum */
  maxh = 0 ;
  for (i = 0 ; i < nbins ; ++i)
    maxh = VL_MAX (maxh, hist [i]) ;

  /* find peaks within 80% from max */
  nangles = 0 ;
  for(i = 0 ; i < nbins ; ++i) {
    double h0 = hist [i] ;
    double hm = hist [(i - 1 + nbins) % nbins] ;
    double hp = hist [(i + 1 + nbins) % nbins] ;

    /* is this a peak? */
    if (h0 > 0.8*maxh && h0 > hm && h0 > hp) {

      /* quadratic interpolation */
      double di = - 0.5 * (hp - hm) / (hp + hm - 2 * h0) ;
      double th = 2 * VL_PI * (i + di + 0.5) / nbins ;
      angles [ nangles++ ] = th ;
      if( nangles == 4 )
        goto enough_angles ;
    }
  }
 enough_angles:
  return nangles ;
}


/** ------------------------------------------------------------------
 ** @internal
 ** @brief Normalizes in norm L_2 a descriptor
 ** @param begin begin of histogram.
 ** @param end   end of histogram.
 **/

VL_INLINE vl_sift_pix
normalize_histogram
(vl_sift_pix *begin, vl_sift_pix *end)
{
  vl_sift_pix* iter ;
  vl_sift_pix  norm = 0.0f ;
#ifdef __wasm_simd128__
  int n = (int)(end - begin) ;
  int i = 0 ;
  v128_t acc0 = wasm_f32x4_splat (0.0f) ;
  v128_t acc1 = wasm_f32x4_splat (0.0f) ;
  for (; i + 8 <= n ; i += 8) {
    v128_t v0 = wasm_v128_load (begin + i) ;
    v128_t v1 = wasm_v128_load (begin + i + 4) ;
    acc0 = wasm_f32x4_add (acc0, wasm_f32x4_mul (v0, v0)) ;
    acc1 = wasm_f32x4_add (acc1, wasm_f32x4_mul (v1, v1)) ;
  }
  v128_t s = wasm_f32x4_add (acc0, acc1) ;
  s = wasm_f32x4_add (s, wasm_i32x4_shuffle (s, s, 2, 3, 0, 1)) ;
  s = wasm_f32x4_add (s, wasm_i32x4_shuffle (s, s, 1, 0, 3, 2)) ;
  norm = wasm_f32x4_extract_lane (s, 0) ;
  for (; i < n ; ++i) norm += begin[i] * begin[i] ;

  norm = vl_sqrt_f (norm) + VL_EPSILON_F ;
  {
    float inv = 1.0f / norm ;
    v128_t vinv = wasm_f32x4_splat (inv) ;
    i = 0 ;
    for (; i + 8 <= n ; i += 8) {
      wasm_v128_store (begin + i,     wasm_f32x4_mul (wasm_v128_load (begin + i),     vinv)) ;
      wasm_v128_store (begin + i + 4, wasm_f32x4_mul (wasm_v128_load (begin + i + 4), vinv)) ;
    }
    for (; i < n ; ++i) begin[i] *= inv ;
  }
  return norm ;
#else
  for (iter = begin ; iter != end ; ++ iter)
    norm += (*iter) * (*iter) ;

  norm = vl_fast_sqrt_f (norm) + VL_EPSILON_F ;

  for (iter = begin; iter != end ; ++ iter)
    *iter /= norm ;

  return norm;
#endif
}

/** ------------------------------------------------------------------
 ** @brief Run the SIFT descriptor on raw data
 **
 ** @param f        SIFT filter.
 ** @param grad     image gradients.
 ** @param descr    SIFT descriptor (output).
 ** @param width    image width.
 ** @param height   image height.
 ** @param x        keypoint x coordinate.
 ** @param y        keypoint y coordinate.
 ** @param sigma    keypoint scale.
 ** @param angle0   keypoint orientation.
 **
 ** The function runs the SIFT descriptor on raw data. Here @a image
 ** is a 2 x @a width x @a height array (by convention, the memory
 ** layout is a s such the first index is the fastest varying
 ** one). The first @a width x @a height layer of the array contains
 ** the gradient magnitude and the second the gradient angle (in
 ** radians, between 0 and @f$ 2\pi @f$). @a x, @a y and @a sigma give
 ** the keypoint center and scale respectively.
 **
 ** In order to be equivalent to a standard SIFT descriptor the image
 ** gradient must be computed at a smoothing level equal to the scale
 ** of the keypoint. In practice, the actual SIFT algorithm makes the
 ** following additional approximation, which influence the result:
 **
 ** - Scale is discretized in @c S levels.
 ** - The image is downsampled once for each octave (if you do this,
 **   the parameters @a x, @a y and @a sigma must be
 **   scaled too).
 **/

VL_EXPORT
void
vl_sift_calc_raw_descriptor (VlSiftFilt const *f,
                             vl_sift_pix const* grad,
                             vl_sift_pix *descr,
                             int width, int height,
                             double x, double y,
                             double sigma,
                             double angle0)
{
  double const magnif = f-> magnif ;

  int          w      = width ;
  int          h      = height ;
  int const    xo     = 2 ;         /* x-stride */
  int const    yo     = 2 * w ;     /* y-stride */

  int          xi     = (int) (x + 0.5) ;
  int          yi     = (int) (y + 0.5) ;

  double const st0    = sin (angle0) ;
  double const ct0    = cos (angle0) ;
  double const SBP    = magnif * sigma + VL_EPSILON_D ;
  int    const W      = floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  int const binto = 1 ;          /* bin theta-stride */
  int const binyo = NBO * NBP ;  /* bin y-stride */
  int const binxo = NBO ;        /* bin x-stride */

  int bin, dxi, dyi ;
  vl_sift_pix const *pt ;
  vl_sift_pix       *dpt ;

  /* check bounds */
  if(xi    <  0               ||
     xi    >= w               ||
     yi    <  0               ||
     yi    >= h -    1        )
    return ;

  /* clear descriptor */
  memset (descr, 0, sizeof(vl_sift_pix) * NBO*NBP*NBP) ;

  /* Center the scale space and the descriptor on the current keypoint.
   * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
   */
  pt  = grad + xi*xo + yi*yo ;
  dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo ;

#undef atd
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   */
  for(dyi =  VL_MAX(- W,   - yi   ) ;
      dyi <= VL_MIN(+ W, h - yi -1) ; ++ dyi) {

    for(dxi =  VL_MAX(- W,   - xi   ) ;
        dxi <= VL_MIN(+ W, w - xi -1) ; ++ dxi) {

      /* retrieve */
      vl_sift_pix mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      vl_sift_pix angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
      vl_sift_pix theta = vl_mod_2pi_f (angle - angle0) ;

      /* fractional displacement */
      vl_sift_pix dx = xi + dxi - x;
      vl_sift_pix dy = yi + dyi - y;

      /* get the displacement normalized w.r.t. the keypoint
         orientation and extension */
      vl_sift_pix nx = ( ct0 * dx + st0 * dy) / SBP ;
      vl_sift_pix ny = (-st0 * dx + ct0 * dy) / SBP ;
      vl_sift_pix nt = NBO * theta / (2 * VL_PI) ;

      /* Get the Gaussian weight of the sample. The Gaussian window
       * has a standard deviation equal to NBP/2. Note that dx and dy
       * are in the normalized frame, so that -NBP/2 <= dx <=
       * NBP/2. */
      vl_sift_pix const wsigma = f->windowSize ;
      vl_sift_pix win = fast_expn
        ((nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;

      /* The sample will be distributed in 8 adjacent bins.
         We start from the ``lower-left'' bin. */
      int         binx = (int)vl_floor_f (nx - 0.5) ;
      int         biny = (int)vl_floor_f (ny - 0.5) ;
      int         bint = (int)vl_floor_f (nt) ;
      vl_sift_pix rbinx = nx - (binx + 0.5) ;
      vl_sift_pix rbiny = ny - (biny + 0.5) ;
      vl_sift_pix rbint = nt - bint ;
      int         dbinx ;
      int         dbiny ;
      int         dbint ;

      /* Distribute the current sample into the 8 adjacent bins*/
      for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
        for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
          for(dbint = 0 ; dbint < 2 ; ++dbint) {

            if (binx + dbinx >= - (NBP/2) &&
                binx + dbinx <    (NBP/2) &&
                biny + dbiny >= - (NBP/2) &&
                biny + dbiny <    (NBP/2) ) {
              vl_sift_pix weight = win
                * mod
                * vl_abs_f (1 - dbinx - rbinx)
                * vl_abs_f (1 - dbiny - rbiny)
                * vl_abs_f (1 - dbint - rbint) ;

              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            }
          }
        }
      }
    }
  }

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if(1) {

    /* normalize L2 norm */
    vl_sift_pix norm = normalize_histogram (descr, descr + NBO*NBP*NBP) ;

    /*
       Set the descriptor to zero if it is lower than our
       norm_threshold.  We divide by the number of samples in the
       descriptor region because the Gaussian window used in the
       calculation of the descriptor is not normalized.
     */
    int numSamples =
      (VL_MIN(W, w - xi -1) - VL_MAX(-W, - xi) + 1) *
      (VL_MIN(W, h - yi -1) - VL_MAX(-W, - yi) + 1) ;

    if(f-> norm_thresh && norm < f-> norm_thresh * numSamples) {
        for(bin = 0; bin < NBO*NBP*NBP ; ++ bin)
            descr [bin] = 0;
    }
    else {
      /* truncate at 0.2. */
      for(bin = 0; bin < NBO*NBP*NBP ; ++ bin) {
        if (descr [bin] > 0.2) descr [bin] = 0.2;
      }

      /* normalize again. */
      normalize_histogram (descr, descr + NBO*NBP*NBP) ;
    }
  }
}

/** ------------------------------------------------------------------
 ** @brief Compute the descriptor of a keypoint
 **
 ** @param f        SIFT filter.
 ** @param descr    SIFT descriptor (output)
 ** @param k        keypoint.
 ** @param angle0   keypoint direction.
 **
 ** The function computes the SIFT descriptor of the keypoint @a k of
 ** orientation @a angle0. The function fills the buffer @a descr
 ** which must be large enough to hold the descriptor.
 **
 ** The function assumes that the keypoint is on the current octave.
 ** If not, it does not do anything.
 **/

VL_EXPORT
void
vl_sift_calc_keypoint_descriptor (VlSiftFilt *f,
                                  vl_sift_pix *descr,
                                  VlSiftKeypoint const* k,
                                  double angle0)
{
  /*
     The SIFT descriptor is a three dimensional histogram of the
     position and orientation of the gradient.  There are NBP bins for
     each spatial dimension and NBO bins for the orientation dimension,
     for a total of NBP x NBP x NBO bins.

     The support of each spatial bin has an extension of SBP = 3sigma
     pixels, where sigma is the scale of the keypoint.  Thus all the
     bins together have a support SBP x NBP pixels wide. Since
     weighting and interpolation of pixel is used, the support extends
     by another half bin. Therefore, the support is a square window of
     SBP x (NBP + 1) pixels. Finally, since the patch can be
     arbitrarily rotated, we need to consider a window 2W += sqrt(2) x
     SBP x (NBP + 1) pixels wide.
  */

  double const magnif      = f-> magnif ;

  double       xper        = pow (2.0, f->o_cur) ;

  int          w           = f-> octave_width ;
  int          h           = f-> octave_height ;
  int const    xo          = 2 ;         /* x-stride */
  int const    yo          = 2 * w ;     /* y-stride */
  int const    so          = 2 * w * h ; /* s-stride */
  double       x           = k-> x     / xper ;
  double       y           = k-> y     / xper ;
  double       sigma       = k-> sigma / xper ;

  int          xi          = (int) (x + 0.5) ;
  int          yi          = (int) (y + 0.5) ;
  int          si          = k-> is ;

  double const st0         = sin (angle0) ;
  double const ct0         = cos (angle0) ;
  double const SBP         = magnif * sigma + VL_EPSILON_D ;
  int    const W           = floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  int const binto = 1 ;          /* bin theta-stride */
  int const binyo = NBO * NBP ;  /* bin y-stride */
  int const binxo = NBO ;        /* bin x-stride */

  int bin, dxi, dyi ;
  vl_sift_pix const *pt ;
  vl_sift_pix       *dpt ;

  /* check bounds */
  if(k->o  != f->o_cur        ||
     xi    <  0               ||
     xi    >= w               ||
     yi    <  0               ||
     yi    >= h -    1        ||
     si    <  f->s_min + 1    ||
     si    >  f->s_max - 2     )
    return ;

  /* synchronize gradient buffer */
  update_gradient (f) ;

  /* VL_PRINTF("W = %d ; magnif = %g ; SBP = %g\n", W,magnif,SBP) ; */

  /* clear descriptor */
  memset (descr, 0, sizeof(vl_sift_pix) * NBO*NBP*NBP) ;

  /* Center the scale space and the descriptor on the current keypoint. */
  pt  = f->grad + xi*xo + yi*yo + (si - f->s_min - 1)*so ;

  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   * Use a padded 6x6x8 histogram so binx,biny in [-3, 1] splat branchlessly.
   */
  {
    float hbuf[6 * 6 * 9] ; /* stride-9 orientation avoids (t0+1)&7 modulo in scatter loop */
    int const dy_min = VL_MAX (- W, 1 - yi    ) ;
    int const dy_max = VL_MIN (+ W, h - yi - 2) ;
    int const dx_min = VL_MAX (- W, 1 - xi    ) ;
    int const dx_max = VL_MIN (+ W, w - xi - 2) ;
    float const inv_SBP      = (float)(1.0 / SBP) ;
    float const ct0_sbp      = (float)(ct0 * inv_SBP) ;
    float const st0_sbp      = (float)(st0 * inv_SBP) ;
    float const wsigma       = (float)f->windowSize ;
    float const inv_2wsigma2 = (float)(1.0 / (2.0 * wsigma * wsigma)) ;
    float const two_pi       = (float)(2.0 * VL_PI) ;
    float const nbo_over_2pi = (float)(NBO / (2.0 * VL_PI)) ;
    float const ang0f        = (float)angle0 ;
    float const base_dx      = (float)(xi - x) ;
    float const base_dy      = (float)(yi - y) ;
    int by, bx ;

    memset (hbuf, 0, sizeof(hbuf)) ;

    for (dyi = dy_min ; dyi <= dy_max ; ++dyi) {
      float dy = base_dy + (float)dyi ;
      float ax = ct0_sbp * base_dx + st0_sbp * dy ;
      float ay = -st0_sbp * base_dx + ct0_sbp * dy ;
      float lo = (float)dx_min ;
      float hi = (float)dx_max ;
      int x_lo, x_hi ;
      float nx, ny ;
      vl_sift_pix const *row ;

      /* Intersect active dxi interval with -2.5 <= nx < 2.5 and -2.5 <= ny < 2.5 */
      if (ct0_sbp > 1e-7f) {
        float l = (-2.5f - ax) / ct0_sbp ;
        float r = ( 2.5f - ax) / ct0_sbp ;
        if (l > lo) lo = l ;
        if (r < hi) hi = r ;
      } else if (ct0_sbp < -1e-7f) {
        float l = ( 2.5f - ax) / ct0_sbp ;
        float r = (-2.5f - ax) / ct0_sbp ;
        if (l > lo) lo = l ;
        if (r < hi) hi = r ;
      } else if (ax < -2.5f || ax >= 2.5f) {
        continue ;
      }

      {
        float sy = -st0_sbp ;
        if (sy > 1e-7f) {
          float l = (-2.5f - ay) / sy ;
          float r = ( 2.5f - ay) / sy ;
          if (l > lo) lo = l ;
          if (r < hi) hi = r ;
        } else if (sy < -1e-7f) {
          float l = ( 2.5f - ay) / sy ;
          float r = (-2.5f - ay) / sy ;
          if (l > lo) lo = l ;
          if (r < hi) hi = r ;
        } else if (ay < -2.5f || ay >= 2.5f) {
          continue ;
        }
      }

      if (lo > hi) continue ;
      x_lo = (int)vl_floor_f (lo) ;
      x_hi = (int)vl_ceil_f (hi) ;
      if (x_lo < dx_min) x_lo = dx_min ;
      if (x_hi > dx_max) x_hi = dx_max ;

      while (x_lo <= x_hi) {
        float tnx = ax + ct0_sbp * (float)x_lo ;
        float tny = ay - st0_sbp * (float)x_lo ;
        if (tnx >= -2.5f && tnx < 2.5f && tny >= -2.5f && tny < 2.5f) break ;
        ++x_lo ;
      }
      while (x_hi >= x_lo) {
        float tnx = ax + ct0_sbp * (float)x_hi ;
        float tny = ay - st0_sbp * (float)x_hi ;
        if (tnx >= -2.5f && tnx < 2.5f && tny >= -2.5f && tny < 2.5f) break ;
        --x_hi ;
      }
      if (x_lo > x_hi) continue ;

      nx = ax + ct0_sbp * (float)x_lo ;
      ny = ay - st0_sbp * (float)x_lo ;
      row = pt + dyi * yo + x_lo * xo ;
      dxi = x_lo ;

#ifdef __wasm_simd128__
      {
        v128_t const vstep_x = wasm_f32x4_splat (4.0f * ct0_sbp) ;
        v128_t const vstep_y = wasm_f32x4_splat (-4.0f * st0_sbp) ;
        v128_t const voff0123 = wasm_f32x4_make (0.0f, 1.0f, 2.0f, 3.0f) ;
        v128_t vnx = wasm_f32x4_add (wasm_f32x4_splat (nx), wasm_f32x4_mul (wasm_f32x4_splat (ct0_sbp), voff0123)) ;
        v128_t vny = wasm_f32x4_sub (wasm_f32x4_splat (ny), wasm_f32x4_mul (wasm_f32x4_splat (st0_sbp), voff0123)) ;
        v128_t const vang0 = wasm_f32x4_splat (ang0f) ;
        v128_t const v2pi = wasm_f32x4_splat (two_pi) ;
        v128_t const vzero = wasm_f32x4_splat (0.0f) ;
        v128_t const vone = wasm_f32x4_splat (1.0f) ;
        v128_t const vhalf = wasm_f32x4_splat (0.5f) ;
        v128_t const v_nbo2pi = wasm_f32x4_splat (nbo_over_2pi) ;
        v128_t const v_inv2ws2_half = wasm_f32x4_splat (-0.5f * inv_2wsigma2) ;
        v128_t const vc2 = wasm_f32x4_splat (0.5f) ;
        v128_t const vc3 = wasm_f32x4_splat (0.16666667f) ;
        v128_t const vc4 = wasm_f32x4_splat (0.04166667f) ;
        v128_t const vc5 = wasm_f32x4_splat (0.00833333f) ;
        v128_t const vthree_i = wasm_i32x4_splat (3) ;
        v128_t const v54_i = wasm_i32x4_splat (54) ;
        v128_t const v9_i = wasm_i32x4_splat (9) ;
        v128_t const v7_i = wasm_i32x4_splat (7) ;

        for (; dxi + 4 <= x_hi + 1 ; dxi += 4, row += 8) {
          v128_t g0 = wasm_v128_load (row) ;
          v128_t g1 = wasm_v128_load (row + 4) ;
          v128_t vmod = wasm_i32x4_shuffle (g0, g1, 0, 2, 4, 6) ;
          v128_t vang = wasm_i32x4_shuffle (g0, g1, 1, 3, 5, 7) ;

          v128_t vtheta = wasm_f32x4_sub (vang, vang0) ;
          vtheta = wasm_v128_bitselect (wasm_f32x4_add (vtheta, v2pi), vtheta, wasm_f32x4_lt (vtheta, vzero)) ;
          vtheta = wasm_v128_bitselect (wasm_f32x4_sub (vtheta, v2pi), vtheta, wasm_f32x4_ge (vtheta, v2pi)) ;
          vtheta = wasm_v128_bitselect (wasm_f32x4_add (vtheta, v2pi), vtheta, wasm_f32x4_lt (vtheta, vzero)) ;
          v128_t vnt = wasm_f32x4_mul (vtheta, v_nbo2pi) ;

          /* exp(-(nx^2+ny^2)/(2*wsigma^2)) via squared degree-5 Taylor on u = -0.5*x in [-0.8, 0] */
          v128_t vu = wasm_f32x4_mul (wasm_f32x4_add (wasm_f32x4_mul (vnx, vnx), wasm_f32x4_mul (vny, vny)), v_inv2ws2_half) ;
          v128_t ve = wasm_f32x4_add (vc4, wasm_f32x4_mul (vu, vc5)) ;
          ve = wasm_f32x4_add (vc3, wasm_f32x4_mul (vu, ve)) ;
          ve = wasm_f32x4_add (vc2, wasm_f32x4_mul (vu, ve)) ;
          ve = wasm_f32x4_add (vone, wasm_f32x4_mul (vu, ve)) ;
          ve = wasm_f32x4_add (vone, wasm_f32x4_mul (vu, ve)) ;
          v128_t vwm = wasm_f32x4_mul (wasm_f32x4_mul (ve, ve), vmod) ;

          v128_t vnxm = wasm_f32x4_sub (vnx, vhalf) ;
          v128_t vnym = wasm_f32x4_sub (vny, vhalf) ;
          v128_t vflx = wasm_f32x4_floor (vnxm) ;
          v128_t vfly = wasm_f32x4_floor (vnym) ;
          v128_t vflt = wasm_f32x4_floor (vnt) ;

          v128_t vrx = wasm_f32x4_sub (vnxm, vflx) ;
          v128_t vry = wasm_f32x4_sub (vnym, vfly) ;
          v128_t vrt = wasm_f32x4_sub (vnt, vflt) ;
          v128_t vwt0 = wasm_f32x4_sub (vone, vrt) ;

          v128_t vwy0 = wasm_f32x4_mul (vwm, wasm_f32x4_sub (vone, vry)) ;
          v128_t vwy1 = wasm_f32x4_mul (vwm, vry) ;
          v128_t vwx0 = wasm_f32x4_sub (vone, vrx) ;

          v128_t vw00 = wasm_f32x4_mul (vwy0, vwx0) ;
          v128_t vw10 = wasm_f32x4_mul (vwy0, vrx) ;
          v128_t vw01 = wasm_f32x4_mul (vwy1, vwx0) ;
          v128_t vw11 = wasm_f32x4_mul (vwy1, vrx) ;

          v128_t vw00_0 = wasm_f32x4_mul (vw00, vwt0) ;
          v128_t vw00_1 = wasm_f32x4_mul (vw00, vrt) ;
          v128_t vw10_0 = wasm_f32x4_mul (vw10, vwt0) ;
          v128_t vw10_1 = wasm_f32x4_mul (vw10, vrt) ;
          v128_t vw01_0 = wasm_f32x4_mul (vw01, vwt0) ;
          v128_t vw01_1 = wasm_f32x4_mul (vw01, vrt) ;
          v128_t vw11_0 = wasm_f32x4_mul (vw11, vwt0) ;
          v128_t vw11_1 = wasm_f32x4_mul (vw11, vrt) ;

          v128_t vbx = wasm_i32x4_add (wasm_i32x4_trunc_sat_f32x4 (vflx), vthree_i) ;
          v128_t vby = wasm_i32x4_add (wasm_i32x4_trunc_sat_f32x4 (vfly), vthree_i) ;
          vbx = wasm_i32x4_min (wasm_i32x4_max (vbx, wasm_i32x4_splat (0)), wasm_i32x4_splat (4)) ;
          vby = wasm_i32x4_min (wasm_i32x4_max (vby, wasm_i32x4_splat (0)), wasm_i32x4_splat (4)) ;
          v128_t vbt0 = wasm_v128_and (wasm_i32x4_trunc_sat_f32x4 (vflt), v7_i) ;
          v128_t voff = wasm_i32x4_add (wasm_i32x4_add (wasm_i32x4_mul (vby, v54_i), wasm_i32x4_mul (vbx, v9_i)), vbt0) ;

          int offs[4] ;
          float c00a[4], c00b[4], c10a[4], c10b[4], c01a[4], c01b[4], c11a[4], c11b[4] ;
          wasm_v128_store (offs, voff) ;
          wasm_v128_store (c00a, vw00_0) ; wasm_v128_store (c00b, vw00_1) ;
          wasm_v128_store (c10a, vw10_0) ; wasm_v128_store (c10b, vw10_1) ;
          wasm_v128_store (c01a, vw01_0) ; wasm_v128_store (c01b, vw01_1) ;
          wasm_v128_store (c11a, vw11_0) ; wasm_v128_store (c11b, vw11_1) ;

          /* Scatter in order 0,2,1,3 to reduce store-to-load forwarding stalls on shared bins */
          {
            static int const ord[4] = {0, 2, 1, 3} ;
            int k ;
            for (k = 0 ; k < 4 ; ++k) {
              int m = ord[k] ;
              float *p0 = hbuf + offs[m] ;
              float *p1 = p0 + 54 ;
              p0[0]  += c00a[m] ; p0[1]  += c00b[m] ;
              p0[9]  += c10a[m] ; p0[10] += c10b[m] ;
              p1[0]  += c01a[m] ; p1[1]  += c01b[m] ;
              p1[9]  += c11a[m] ; p1[10] += c11b[m] ;
            }
          }

          vnx = wasm_f32x4_add (vnx, vstep_x) ;
          vny = wasm_f32x4_add (vny, vstep_y) ;
        }
        nx = ax + ct0_sbp * (float)dxi ;
        ny = ay - st0_sbp * (float)dxi ;
      }
#endif

      for (; dxi <= x_hi ; ++dxi, nx += ct0_sbp, ny -= st0_sbp, row += xo) {
        int binx, biny, bint, t0 ;
        float rbinx, rbiny, rbint, wx0, wx1, wy0, wy1, wt0, wt1, win, mod, angle, theta, nt, wm ;
        float *p0, *p1 ;

        mod   = row[0] ;
        angle = row[1] ;
        theta = angle - ang0f ;
        if (theta < 0.0f) {
          theta += two_pi ;
          if (theta < 0.0f) theta = vl_mod_2pi_f (theta) ;
        } else if (theta >= two_pi) {
          theta -= two_pi ;
          if (theta >= two_pi) theta = vl_mod_2pi_f (theta) ;
        }

        nt  = theta * nbo_over_2pi ;
        win = (float)fast_expn ((double)((nx * nx + ny * ny) * inv_2wsigma2)) ;
        wm  = win * mod ;

        binx = (int)vl_floor_f (nx - 0.5f) ;
        biny = (int)vl_floor_f (ny - 0.5f) ;
        bint = (int)vl_floor_f (nt) ;
        rbinx = nx - (binx + 0.5f) ;
        rbiny = ny - (biny + 0.5f) ;
        rbint = nt - bint ;

        wx1 = rbinx ; wx0 = 1.0f - rbinx ;
        wy1 = rbiny ; wy0 = 1.0f - rbiny ;
        wt1 = rbint ; wt0 = 1.0f - rbint ;

        t0 = bint & (NBO - 1) ;
        p0 = hbuf + (biny + 3) * 54 + (binx + 3) * 9 + t0 ;
        p1 = p0 + 54 ;
        {
          float wmy0 = wm * wy0 ;
          float wmy1 = wm * wy1 ;
          float w00 = wmy0 * wx0, w10 = wmy0 * wx1 ;
          float w01 = wmy1 * wx0, w11 = wmy1 * wx1 ;
          p0[0]  += w00 * wt0 ; p0[1]  += w00 * wt1 ;
          p0[9]  += w10 * wt0 ; p0[10] += w10 * wt1 ;
          p1[0]  += w01 * wt0 ; p1[1]  += w01 * wt1 ;
          p1[9]  += w11 * wt0 ; p1[10] += w11 * wt1 ;
        }
      }
    }

    /* Fold stride-9 bin[8] back into bin[0] and copy interior 4x4x8 region into descr */
    for (by = 0 ; by < NBP ; ++by) {
      for (bx = 0 ; bx < NBP ; ++bx) {
        float *src = hbuf + (by + 1) * 54 + (bx + 1) * 9 ;
        float *dst = descr + (by * NBP + bx) * NBO ;
        dst[0] = src[0] + src[8] ;
        dst[1] = src[1] ;
        dst[2] = src[2] ;
        dst[3] = src[3] ;
        dst[4] = src[4] ;
        dst[5] = src[5] ;
        dst[6] = src[6] ;
        dst[7] = src[7] ;
      }
    }
  }

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if(1) {

    /* Normalize the histogram to L2 unit length. */
    vl_sift_pix norm = normalize_histogram (descr, descr + NBO*NBP*NBP) ;

    /* Set the descriptor to zero if it is lower than our norm_threshold */
    if(f-> norm_thresh && norm < f-> norm_thresh) {
        for(bin = 0; bin < NBO*NBP*NBP ; ++ bin)
            descr [bin] = 0;
    }
    else {

      /* Truncate at 0.2. */
#ifdef __wasm_simd128__
      {
        v128_t vmax = wasm_f32x4_splat (0.2f) ;
        for (bin = 0 ; bin < NBO*NBP*NBP ; bin += 4) {
          wasm_v128_store (descr + bin, wasm_f32x4_pmin (wasm_v128_load (descr + bin), vmax)) ;
        }
      }
#else
      for(bin = 0; bin < NBO*NBP*NBP ; ++ bin) {
        if (descr [bin] > 0.2) descr [bin] = 0.2;
      }
#endif

      /* Normalize again. */
      normalize_histogram (descr, descr + NBO*NBP*NBP) ;
    }
  }

}

/** ------------------------------------------------------------------
 ** @brief Initialize a keypoint from its position and scale
 **
 ** @param f     SIFT filter.
 ** @param k     SIFT keypoint (output).
 ** @param x     x coordinate of the keypoint center.
 ** @param y     y coordinate of the keypoint center.
 ** @param sigma keypoint scale.
 **
 ** The function initializes a keypoint structure @a k from
 ** the location @a x
 ** and @a y and the scale @a sigma of the keypoint. The keypoint structure
 ** maps the keypoint to an octave and scale level of the discretized
 ** Gaussian scale space, which is required for instance to compute the
 ** keypoint SIFT descriptor.
 **
 ** @par Algorithm
 **
 ** The formula linking the keypoint scale sigma to the octave and
 ** scale indexes is
 **
 ** @f[ \sigma(o,s) = \sigma_0 2^{o+s/S} @f]
 **
 ** In addition to the scale index @e s (which can be fractional due
 ** to scale interpolation) a keypoint has an integer scale index @e
 ** is too (which is the index of the scale level where it was
 ** detected in the DoG scale space). We have the constraints (@ref
 ** sift-tech-detector see also the "SIFT detector"):
 **
 ** - @e o is integer in the range @f$ [o_\mathrm{min},
 **   o_{\mathrm{min}}+O-1] @f$.
 ** - @e is is integer in the range @f$ [s_\mathrm{min}+1,
 **   s_\mathrm{max}-2] @f$.  This depends on how the scale is
 **   determined during detection, and must be so here because
 **   gradients are computed only for this range of scale levels
 **   and are required for the calculation of the SIFT descriptor.
 ** - @f$ |s - is| < 0.5 @f$ for detected keypoints in most cases due
 **   to the interpolation technique used during detection. However
 **   this is not necessary.
 **
 ** Thus octave o represents scales @f$ \{ \sigma(o, s) : s \in
 ** [s_\mathrm{min}+1-.5, s_\mathrm{max}-2+.5] \} @f$. Note that some
 ** scales may be represented more than once. For each scale, we
 ** select the largest possible octave that contains it, i.e.
 **
 ** @f[
 **  o(\sigma)
 **  = \max \{ o \in \mathbb{Z} :
 **    \sigma_0 2^{\frac{s_\mathrm{min}+1-.5}{S}} \leq \sigma \}
 **  = \mathrm{floor}\,\left[
 **    \log_2(\sigma / \sigma_0) - \frac{s_\mathrm{min}+1-.5}{S}\right]
 ** @f]
 **
 ** and then
 **
 ** @f[
 ** s(\sigma) = S  \left[\log_2(\sigma / \sigma_0) - o(\sigma)\right],
 ** \quad
 ** is(\sigma) = \mathrm{round}\,(s(\sigma))
 ** @f]
 **
 ** In practice, both @f$ o(\sigma) @f$ and @f$ is(\sigma) @f$ are
 ** clamped to their feasible range as determined by the SIFT filter
 ** parameters.
 **/

VL_EXPORT
void
vl_sift_keypoint_init (VlSiftFilt const *f,
                       VlSiftKeypoint *k,
                       double x,
                       double y,
                       double sigma)
{
  int    o, ix, iy, is ;
  double s, phi, xper ;

  phi = log2 ((sigma + VL_EPSILON_D) / f->sigma0) ;
  o   = (int)vl_floor_d (phi -  ((double) f->s_min + 0.5) / f->S) ;
  o   = VL_MIN (o, f->o_min + f->O - 1) ;
  o   = VL_MAX (o, f->o_min           ) ;
  s   = f->S * (phi - o) ;

  is  = (int)(s + 0.5) ;
  is  = VL_MIN(is, f->s_max - 2) ;
  is  = VL_MAX(is, f->s_min + 1) ;

  xper = pow (2.0, o) ;
  ix   = (int)(x / xper + 0.5) ;
  iy   = (int)(y / xper + 0.5) ;

  k -> o  = o ;

  k -> ix = ix ;
  k -> iy = iy ;
  k -> is = is ;

  k -> x = x ;
  k -> y = y ;
  k -> s = s ;

  k->sigma = sigma ;
}
