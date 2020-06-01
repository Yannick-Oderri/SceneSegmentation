//
// Created by ynki9 on 5/25/20.
//
#include "utils/cuda_helper.cuh"
__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale)
{
    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short)(fScale*(abs((int)Horz)+abs((int)Vert)));

    if (Sum < 0)
    {
        return 0;
    }
    else if (Sum > 0xff)
    {
        return 0xff;
    }

    return (unsigned char) Sum;
}

__global__
void curveDisc_kernel(float* out_data, int Pitch,
                        int w, int h, cudaTextureObject_t tex){
    float *res_pixel =
            (float *)(((char*)out_data)+(blockIdx.x*Pitch));

    res_pixel[threadIdx.x] = 20;
    int start = threadIdx.x * (w / threadDim.x);
    int end = start + threadDim.x;
    for (int i = threadIdx.x; i < w; i++)
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
        res_pixel[i] = tex2D<float>(tex, i, blockIdx.x);
        res_pixel[i] = ComputeSobel(pix00, pix01, pix02,
                                    pix10, pix11, pix12,
                                    pix20, pix21, pix22, 1);
    }
}

__host__
void t_setupDepthMap(int width, int height, cv::Mat depth_map, cudaTextureObject_t& dev_depth_tex){
    size_t pitch;
    float* dev_dimg;
    uchar* dimg = depth_map.data;

    checkCudaErrors(cudaMallocPitch(&dev_dimg, &pitch, sizeof(float)*width, height));
    checkCudaErrors(cudaMemcpy2D(dev_dimg, pitch, dimg, depth_map.step, sizeof(float)*width,
                                 height, cudaMemcpyHostToDevice));

    /// Intialize Resoure Descriptor
    cudaResourceDesc texRes;
    memset(&texRes, 0x0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr   = dev_dimg;
    texRes.res.pitch2D.desc     = cudaCreateChannelDesc<float>();;
    texRes.res.pitch2D.width    = width;
    texRes.res.pitch2D.height   = height;
    texRes.res.pitch2D.pitchInBytes = pitch;

    /// Initialize Texture Descriptor
    cudaTextureDesc texDescr;
    memset(&texDescr, 0x0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&dev_depth_tex, &texRes, &texDescr, NULL));
}

__host__
cv::Mat launchCurveDiscOprKernel(cv::Mat& depth_map){

    unsigned int height = depth_map.rows;
    unsigned int width = depth_map.cols;

    cudaTextureObject_t depth_tex;
    t_setupDepthMap(width, height, depth_map, depth_tex);

    size_t pitch;
    float *d_out;
    checkCudaErrors(cudaMallocPitch((void**)&d_out, &pitch, sizeof(float)*width, height));
    // cudaMalloc(&d_out, depth_map.rows*depth_map.cols*sizeof(float));

    /// Execute Cuda kernel
    dim3 threadPerBlock(16, 16);
    int blockCount = 480; //depth_map.cols;
    int step = (int)pitch;

    curveDisc_kernel<<<480, 16>>>(d_out, step, width, height, depth_tex);

    /// Get results to opencv mat
    float* h_data = (float*)malloc(pitch * height);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(h_data, d_out, pitch * height, cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_32F, h_data, pitch);

    /// cleanup temp contours
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaDestroyTextureObject(depth_tex));

    return result;
}

extern "C"
cv::Mat cuCurveDiscOperation(cv::Mat& depth_map){
    cv::Mat res = launchCurveDiscOprKernel(depth_map);

    cv::imshow("test", res);
    cv::imwrite("test.png", res);
    cv::waitKey(0);
}