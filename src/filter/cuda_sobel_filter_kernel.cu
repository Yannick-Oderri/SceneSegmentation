#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cuda_sobel_filter_kernel.h"


__device__ float
ComputeSobel(float ul, // upper left
             float um, // upper middle
             float ur, // upper right
             float ml, // middle left
             float mm, // middle (unused)
             float mr, // middle right
             float ll, // lower left
             float lm, // lower middle
             float lr, // lower right
             float fScale)
{
    float Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    float Vert = ul + 2*um + ur - ll - 2*lm - lr;
    float Sum = (fScale * (Horz + Vert));

    return (float) Sum;
}


__global__ void
SobelTex(float* in_data, float* out_data, unsigned int Pitch,
         int w, int h, float fScale, cudaTextureObject_t tex)
{
    float *res_pixel =
            (float *)(((float *) out_data)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        float pix00 = tex2D<float>(tex, (float) i-1, (float) blockIdx.x-1);
        float pix01 = tex2D<float>(tex, (float) i+0, (float) blockIdx.x-1);
        float pix02 = tex2D<float>(tex, (float) i+1, (float) blockIdx.x-1);
        float pix10 = tex2D<float>(tex, (float) i-1, (float) blockIdx.x+0);
        float pix11 = tex2D<float>(tex, (float) i+0, (float) blockIdx.x+0);
        float pix12 = tex2D<float>(tex, (float) i+1, (float) blockIdx.x+0);
        float pix20 = tex2D<float>(tex, (float) i-1, (float) blockIdx.x+1);
        float pix21 = tex2D<float>(tex, (float) i+0, (float) blockIdx.x+1);
        float pix22 = tex2D<float>(tex, (float) i+1, (float) blockIdx.x+1);
        res_pixel[i] = ComputeSobel(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale);
    }
}


// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void sobelFilter(Pixel *in_data, Pixel* out_data, int iw, int ih, float fScale, cudaTextureObject_t tex)
{
    SobelTex<<<ih, iw/2>>>(in_data, out_data, iw, iw, ih, fScale, tex);
}
