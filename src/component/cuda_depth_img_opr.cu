//
// Created by ynki9 on 5/25/20.
//
#include "utils/cuda_helper.cuh"
#include <npp.h>

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

    int height = depth_map.rows;
    int width = depth_map.cols;



    int pitch;
    float *img_data = (float*)depth_map.data;

    Npp32f* dev_img = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* scratch_buffer = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* x_grad = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* y_grad = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* mag_grad = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* ang_grad = nppiMalloc_32f_C1(width, height, &pitch);

    NppiSize src_size = {width, height};
    float channel_noise[] = {0.0000011f};

    //nppiCopy_32f_C1R(img_data, depth_map.step, dev_img, pitch, src_size);
    checkCudaErrors(cudaMemcpy2D(dev_img, pitch, img_data, depth_map.step, sizeof(float)*width,
                                 height, cudaMemcpyHostToDevice));
    // cudaMalloc(&d_out, depth_map.rows*depth_map.cols*sizeof(float));
    NppStatus res;

//    res = nppiFilterWienerBorder_32f_C1R(
//            dev_img,
//            pitch,
//            src_size,
//            {0, 0},
//            scratch_buffer,
//            pitch,
//            src_size,
//            {5, 5},
//            {0, 0},
//            channel_noise,
//            NppiBorderType::NPP_BORDER_REPLICATE
//    );
//
//
//    float* w_data = (float*)malloc(pitch * height);
//    cudaMemcpy(w_data, scratch_buffer, pitch * height, cudaMemcpyDeviceToHost);
//    cv::Mat wwienerRes(height, width, CV_32F, w_data, pitch);
//    cv::Mat ttres;
//    cv::normalize(wwienerRes, ttres, 0, 255, CV_MINMAX, CV_8U);
//    cv::imshow("1st Wiener Filtered", ttres);
//
//
//    Npp32f* t_addr = dev_img;
//    dev_img = scratch_buffer;
//    scratch_buffer = t_addr;

    res = nppiFilterMedian_32f_C1R(
            dev_img,
            pitch,
            scratch_buffer,
            pitch,
            {width, height},
            {5, 5},
            {0, 0},
            reinterpret_cast<Npp8u *>(x_grad)
    );

    Npp32f* t_addr = dev_img;
    dev_img = scratch_buffer;
    scratch_buffer = t_addr;

    res = nppiGradientVectorSobelBorder_32f_C1R(
            dev_img,
            pitch,
            src_size,
            {0,0},
            x_grad,
            pitch,
            y_grad,
            pitch,
            mag_grad,
            pitch,
            ang_grad,
            pitch, {640, 480},
            NppiMaskSize::NPP_MASK_SIZE_5_X_5,
            NppiNorm::nppiNormL1,
            NppiBorderType::NPP_BORDER_REPLICATE
            );

    if(res != 0){
        std::cout << "NPPI Error: " << res;
    }

    res = nppiFilterMedian_32f_C1R(
            ang_grad,
            pitch,
            scratch_buffer,
            pitch,
            {width, height},
            {5, 5},
            {3, 3},
            reinterpret_cast<Npp8u *>(x_grad)
            );
//    channel_noise[0] = 0.00100001f;
//    res = nppiFilterWienerBorder_32f_C1R(
//            ang_grad,
//            pitch,
//            src_size,
//            {0, 0},
//            scratch_buffer,
//            pitch,
//            src_size,
//            {12, 12},
//            {0, 0},
//            channel_noise,
//            NppiBorderType::NPP_BORDER_REPLICATE
//    );

    if(res != 0){
        std::cout << "NPPI Error: " << res;
    }

    /// Get results to opencv mat
    float* h_data = (float*)malloc(pitch * height);
    float* w_data = (float*)malloc(pitch * height);

    cudaMemcpy(h_data, ang_grad, pitch * height, cudaMemcpyDeviceToHost);
    cv::Mat result(height, width, CV_32F, h_data, pitch);
    cv::imshow("test", result);

    cudaMemcpy(w_data, scratch_buffer, pitch * height, cudaMemcpyDeviceToHost);
    cv::Mat wienerRes(height, width, CV_32F, w_data, pitch);
    cv::imshow("Wiener Filtered", wienerRes);

    cv::Mat tres;
    cv::normalize(wienerRes, tres, 0, 255, CV_MINMAX, CV_8U);
    cv::Canny(tres, result, 140, 225);


    cv::imshow("canny result", result);
    cv::imwrite("canny.png", result);

    cv::morphologyEx(result, result, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    cv::imshow("morph res", result);

    return result;
}

extern "C"
cv::Mat cuCurveDiscOperation(cv::Mat& depth_map){
    cv::Mat tres = launchCurveDiscOprKernel(depth_map);
    cv::Mat res(tres.size(), CV_32F);
    //cv::normalize(tres, res, 0, 1, CV_MINMAX, CV_32F);
    //cv::normalize(depth_map, depth_map, 0, 1, CV_MINMAX, CV_32F);

    cv::imshow("Initial Image", depth_map);


    cv::waitKey(0);
    cv::imwrite("test.png", tres);
}