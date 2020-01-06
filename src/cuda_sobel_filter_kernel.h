//
// Created by ynki9 on 12/30/19.
//

#ifndef PROJECT_EDGE_CUDA_SOBEL_FILTER_KERNEL_H
#define PROJECT_EDGE_CUDA_SOBEL_FILTER_KERNEL_H

typedef float Pixel;

// global determines which filter to invoke
enum SobelDisplayMode
{
    SOBELDISPLAY_IMAGE = 0,
    SOBELDISPLAY_SOBELTEX,
    SOBELDISPLAY_SOBELSHARED
};


extern enum SobelDisplayMode g_SobelDisplayMode;

extern "C" void sobelFilter(Pixel *in_data, Pixel* out_data, int iw, int ih, float fScale, cudaTextureObject_t tex);
extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTexture(void);
extern "C" void initFilter(void);


#endif //PROJECT_EDGE_CUDA_SOBEL_FILTER_KERNEL_H
