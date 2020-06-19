//
// Created by ynki9 on 5/25/20.
//
#include "utils/cuda_helper.cuh"
#include <npp.h>

#define DEBUG_CD_OPR_OUTPUT

__global__
void wiener2_noise(cudaTextureObject_t src_img, NppiSize src_size, Npp32f* dst_img, Npp32s dst_step, Npp32f noise){
    Npp32f *row =
            (float *)(((char*)dst_img)+(blockIdx.x*dst_step));

    int l = src_size.width / blockDim.x;
    int b = threadIdx.x * l;

    for (int i = b; i < b + l; i++) {
    }

}
__global__
void wiener2(cudaTextureObject_t src_img, NppiSize src_size, Npp32f* dst_img, Npp32s dst_step, Npp32f noise){
    Npp32f *row =
            (float *)(((char*)dst_img)+(blockIdx.x*dst_step));

    int l = src_size.width / blockDim.x;
    int b = threadIdx.x * l;

    for (int i = b; i < b + l; i++){
        Npp32f local_mean = 0;
        Npp32f local_variance = 0;
        for(int u = 0; u < 5; u++){
            for(int v = 0; v < 5; v++){
                Npp32f pixel = tex2D<float>(src_img, (int) (i+v-2), (int)(blockIdx.x+u-2));
                local_mean = local_mean + pixel;
                local_variance = local_variance + (pixel * pixel);
            }
        }

        local_mean = local_mean / 25.0f;
        local_variance = local_variance / 25.0f;
        local_variance = local_variance - (local_mean * local_mean);

        Npp32f g = tex2D<float>(src_img, (int) i, (int) blockIdx.x);
        Npp32f f = g - local_mean;
        g = local_variance - noise;
        g = fmaxf(g, 0);
        local_variance = fmaxf(local_variance, noise);
        f = f / local_variance;
        f = f * g;
        f = f + local_mean;

        row[i] = f;
    }
}



__global__
void gradientOperation(Npp32f* x_grad, Npp32f* y_grad,
        NppiSize src_size, Npp32f* lr_img, Npp32f* ud_img, Npp32s dst_step, Npp32f grad_tresh){
    Npp32f *x_grad_row =
            (float *)(((char*)x_grad)+(blockIdx.x*dst_step));
    Npp32f *y_grad_row =
            (float *)(((char*)y_grad)+(blockIdx.x*dst_step));

    Npp32f *ud_gdir =
            (float *)(((char*)ud_img)+(blockIdx.x*dst_step));
    Npp32f *lr_gdir =
            (float *)(((char*)lr_img)+(blockIdx.x*dst_step));

    int l = src_size.width / blockDim.x;
    int b = threadIdx.x * l;

    for (int i = b; i < b + l; i++) {
        float x_grad = x_grad_row[i]; //tex2D<float>(x_grad, (float) i, (float) blockIdx.x);
        float y_grad = y_grad_row[i]; //tex2D<float>(y_grad, (float) i, (float) blockIdx.x);
        if (abs(x_grad) < grad_tresh && abs(y_grad) < grad_tresh){
            ud_gdir[i] = 0;
            lr_gdir[i] = 0;
            continue;
        }

        float gdir = atan2(-y_grad, x_grad);
        lr_gdir[i] = gdir;
        ud_gdir[i] = abs(gdir);
        if (gdir > M_PI/2) {
            lr_gdir[i] = M_PI - gdir;
        }else if(gdir < -M_PI/2) {
            lr_gdir[i] = -(M_PI+gdir);
        }
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

void setupCudaTexture(int width, int height, Npp32f* cuda_img, size_t pitch, cudaTextureObject_t& dev_depth_tex){
    /// Intialize Resoure Descriptor
    cudaResourceDesc texRes;
    memset(&texRes, 0x0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr   = cuda_img;
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

#ifdef DEBUG_CD_OPR_OUTPUT
    cv::Mat td_img;
    cv::normalize(depth_map, td_img, 0, 1, CV_MINMAX, CV_32F);
    cv::imshow("CD: Initial Image", td_img);
#endif

    int height = depth_map.rows;
    int width = depth_map.cols;


    cudaTextureObject_t depth_tex;
    t_setupDepthMap(width, height, depth_map, depth_tex);


    int pitch;
    float *img_data = (float*)depth_map.data;

    Npp32f* dev_img = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* sbuffer1 = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* sbuffer2 = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* x_grad = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* y_grad = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* ud_gdir = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* lr_gdir = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* ang_grd = nppiMalloc_32f_C1(width, height, &pitch);
    Npp32f* mag_grd = nppiMalloc_32f_C1(width, height, &pitch);

    NppiSize src_size = {width, height};
    Npp32f img_noise = 4.046285e+04;
    Npp32f xgrad_noise = 4.046285e+04;
    Npp32f ygrad_noise = 4.909884e+05;
    Npp32f lrgdir_noise = 1.81816e+02;
    Npp32f udgdir_noise = 2.24974e+02;

    checkCudaErrors(cudaMemcpy2D(dev_img, pitch, img_data, depth_map.step, sizeof(float)*width,
                                 height, cudaMemcpyHostToDevice));
    // cudaMalloc(&d_out, depth_map.rows*depth_map.cols*sizeof(float));
    NppStatus res;
    Npp32f* d_mean_f; cudaMalloc(&d_mean_f, sizeof(Npp32f));
    Npp32f* d_std_f; cudaMalloc(&d_std_f, sizeof(Npp32f));

    // Filter Input Image
    wiener2<<<height, 16>>>(depth_tex, src_size, sbuffer1, pitch, img_noise);
    checkCudaErrors(cudaDeviceSynchronize());

    #ifdef DEBUG_CD_OPR_OUTPUT
    float* stage1_filter = (float*)malloc(pitch * height);
    cudaMemcpy(stage1_filter, sbuffer1, pitch * height, cudaMemcpyDeviceToHost);
    cv::Mat stage1_res(height, width, CV_32F, stage1_filter, pitch);
    cv::normalize(stage1_res, stage1_res, 0, 1, CV_MINMAX, CV_32F);
    cv::imshow("Stage 1 Filtering", stage1_res);
    #endif

    // Perform Gradient
    res = nppiGradientVectorSobelBorder_32f_C1R(
            sbuffer1,
            pitch,
            src_size,
            {0,0},
            x_grad,
            pitch,
            y_grad,
            pitch,
            mag_grd,
            pitch,
            ang_grd,
            pitch, {640, 480},
            NppiMaskSize::NPP_MASK_SIZE_3_X_3,
            NppiNorm::nppiNormL1,
            NppiBorderType::NPP_BORDER_REPLICATE
            );

    // Filter Gradients
    cudaTextureObject_t tx_grad;
    cudaTextureObject_t ty_grad;
    setupCudaTexture(width, height, x_grad, pitch, tx_grad);
    setupCudaTexture(width, height, y_grad, pitch, ty_grad);
    wiener2<<<height, 16>>>(tx_grad, src_size, sbuffer1, pitch, xgrad_noise);
    wiener2<<<height, 16>>>(ty_grad, src_size, sbuffer2, pitch, ygrad_noise);


#ifdef DEBUG_CD_OPR_OUTPUT
    float* stage2_xgrad = (float*)malloc(pitch * height);
    cudaMemcpy(stage2_xgrad, sbuffer1, pitch * height, cudaMemcpyDeviceToHost);
    cv::Mat stage2_xres(height, width, CV_32F, stage2_xgrad, pitch);
    cv::normalize(stage2_xres, stage2_xres, 0, 1, CV_MINMAX, CV_32F);
    cv::imshow("Stage 2 X Gradient", stage2_xres);

    float* stage2_ygrad = (float*)malloc(pitch * height);
    cudaMemcpy(stage2_ygrad, sbuffer2, pitch * height, cudaMemcpyDeviceToHost);
    cv::Mat stage2_yres(height, width, CV_32F, stage2_ygrad, pitch);
    cv::normalize(stage2_yres, stage2_yres, 0, 1, CV_MINMAX, CV_32F);
    cv::imshow("Stage 2 Y Gradient", stage2_yres);
#endif

    // Perform Gradient Mirroring
    gradientOperation<<<height, 16>>>(sbuffer1, sbuffer2, src_size, lr_gdir, ud_gdir, pitch, 2.0);
    cudaTextureObject_t wtx_grad;
    cudaTextureObject_t wty_grad;
    setupCudaTexture(width, height, lr_gdir, pitch, wtx_grad);
    setupCudaTexture(width, height, ud_gdir, pitch, wty_grad);
    wiener2<<<height, 16>>>(wtx_grad, src_size, sbuffer1, pitch, lrgdir_noise);
    wiener2<<<height, 16>>>(wty_grad, src_size, sbuffer2, pitch, udgdir_noise);

    float* stage3_lrgdir_res = (float*)malloc(pitch * height);
    cudaMemcpy(stage3_lrgdir_res, sbuffer1, pitch * height, cudaMemcpyDeviceToHost);
    cv::Mat stage3_lrgdir_img(height, width, CV_32F, stage3_lrgdir_res, pitch);
    cv::normalize(stage3_lrgdir_img, stage3_lrgdir_img, 0, 255, CV_MINMAX, CV_8U);

    float* stage3_udgdir_res = (float*)malloc(pitch * height);
    cudaMemcpy(stage3_udgdir_res, sbuffer2, pitch * height, cudaMemcpyDeviceToHost);
    cv::Mat stage3_udgdir_img(height, width, CV_32F, stage3_udgdir_res, pitch);
    cv::normalize(stage3_udgdir_img, stage3_udgdir_img, 0, 255, CV_MINMAX, CV_8U);


#ifdef DEBUG_CD_OPR_OUTPUT
    cv::imshow("Stage 3 LR Grad Mirroring", stage3_lrgdir_img);
    cv::imshow("Stage 3 UD Grad Mirroring", stage3_udgdir_img);
#endif

    float highThresh = 0.75;
    float lowThresh = 0.4 * highThresh;

    cv::Mat hori_edge(height, width, CV_8U);
    cv::Canny(stage3_lrgdir_img, hori_edge, 255*lowThresh, 255*highThresh);

    cv::Mat vert_edge(height, width, CV_8U);
    cv::Canny(stage3_udgdir_img, vert_edge, 255*lowThresh, 255*highThresh);

    cv::Mat result = hori_edge | vert_edge;

#ifdef DEBUG_CD_OPR_OUTPUT
    cv::imshow("Stage 4 CD Hori Results", hori_edge);
    cv::imshow("Stage 4 CD Vert Results", vert_edge);
    cv::imwrite("cd_res.png", result);
    cv::imshow("Final CD Results", result);
#endif


    (nppiFree(sbuffer1));
    (nppiFree(sbuffer2));
    (nppiFree(ang_grd));
    (nppiFree(mag_grd));
    (nppiFree(sbuffer2));
    checkCudaErrors(cudaDestroyTextureObject(depth_tex));
    checkCudaErrors(cudaDestroyTextureObject(tx_grad));
    checkCudaErrors(cudaDestroyTextureObject(ty_grad));
    checkCudaErrors(cudaDestroyTextureObject(wtx_grad));
    checkCudaErrors(cudaDestroyTextureObject(wty_grad));

    return result;
}

/// Shrink LUTS
const unsigned char LUT_shrink1[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0
};

const unsigned char LUT_shrink2[] = {
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,1,0,0,1,0,1,1,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

/// Skel LUTS
const unsigned char LUT_skel1[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,0,1,1,1,1,0
};

const unsigned char LUT_skel2[] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,0,0,1,0,1,1,0,0,0,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};


/// Bridge Luts
const unsigned char LUT_bridge1[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

// HBreak LUTS
unsigned char LUT_hbreak1[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

// Spurs Luts
unsigned char LUT_spur1[] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

/// Thin LUTs

unsigned char LUT_thin1[] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1
};

unsigned char LUT_thin2[] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};


void bwlookup( const cv::Mat & in, cv::Mat & out, const cv::Mat & lut, int bordertype=cv::BORDER_CONSTANT, cv::Scalar px = cv::Scalar(0) )
{
    if ( in.type() != CV_8UC1 )
        CV_Error(CV_StsError, "er");
    if ( lut.type() != CV_8UC1 || lut.rows*lut.cols!=512 || !lut.isContinuous() )
        CV_Error(CV_StsError, "lut size != 512" );
    if ( out.type() != in.type() || out.size() != in.size() )
        out = cv::Mat( in.size(), in.type() );

    const unsigned char * _lut = lut.data;
    cv::Mat t;
    cv::copyMakeBorder( in,t,1,1,1,1,bordertype,px);
    const int rows=in.rows+1;
    const int cols=in.cols+1;
    for ( int y=1;y<rows;++y)
    {
        for ( int x=1;x<cols;++x)
        {
            int L = 0;
            const int jmax=y+1;
#if 1 // row-major order
            for ( int j=y-1, k=1; j<=jmax; ++j, k<<=3 )
			{
				const unsigned char * p = t.ptr<unsigned char>(j) + x-1;
				for ( unsigned int u=0;u<3;++u )
				{
					if ( p[u] )
						L += (k<<u);
                }
            }
#else // column-major order (MATLAB)
            for ( int j=y-1, k=1; j<=jmax; ++j, k<<=1 )
            {
                const unsigned char * p = t.ptr<unsigned char>(j) + x-1;
                for ( unsigned int u=0;u<3;++u )
                {
                    if ( p[u] )
                        L += (k<<3*u);

                }
            }
#endif
            out.at<unsigned char>(y-1,x-1)=_lut[ L ];
        }
    }
}

cv::Mat skeletonize(cv::Mat& in, int n){
    cv::Mat scratch1 = in / 255;
    cv::Mat morph_out;
    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        cv::Mat x = (scratch1 != 0) / 255;
        // Perform Morphological skeletonization
        cv::Mat lut(1, 512, CV_8UC1, (void *) LUT_skel1);
        bwlookup(scratch1, morph_out, lut);

        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_skel2);
        bwlookup(morph_out, scratch1, lut);
        morph_out = x & scratch1;

        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_bridge1);
        bwlookup(morph_out, scratch1, lut);
    }

    return scratch1;
}

cv::Mat shrink(cv::Mat& in, int n){
    cv::Mat scratch1 = in.clone();
    cv::Mat morph_out;
    cv::Mat lut;
    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        // Perform Shrink
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_shrink1);
        bwlookup(scratch1, morph_out, lut);
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_shrink2);
        bwlookup(morph_out, scratch1, lut);
        scratch1 &= in;
    }

    return scratch1;
}

cv::Mat clean(cv::Mat& in){
    const int CONECTIVITY = 8;
    cv::Mat morph;
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    kernel.at<unsigned char>(1, 1) = CONECTIVITY;
    cv::filter2D(in, morph, -1, kernel);

    return morph > CONECTIVITY;
}

cv::Mat hbreak(cv::Mat& in, int n){
    cv::Mat scratch1 = in.clone();
    cv::Mat morph_out;
    cv::Mat lut;

    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        // Perform hbreak
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_hbreak1);
        bwlookup(scratch1, morph_out, lut);
        scratch1 = morph_out.clone();
    }
    return scratch1;
}

cv::Mat spur(cv::Mat& in, int n){
    cv::Mat scratch1 = in.clone();
    cv::Mat morph_out;
    cv::Mat lut;

    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        // Perform hbreak
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_spur1);
        bwlookup(scratch1, morph_out, lut);
        scratch1 = morph_out.clone();
    }
    return scratch1;
}

cv::Mat thin(cv::Mat& in, int n){
    cv::Mat scratch1 = in.clone();
    cv::Mat morph_out;
    cv::Mat lut;
    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        // Perform Shrink
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_thin1);
        bwlookup(scratch1, morph_out, lut);
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_thin2);
        bwlookup(morph_out, scratch1, lut);
    }

    return scratch1;
}

extern "C"
cv::Mat cuCurveDiscOperation(cv::Mat& depth_map){
    cv::Mat x = launchCurveDiscOprKernel(depth_map);
    cv::morphologyEx(x, x, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    cv::Mat img = skeletonize(x, 15);
    img = shrink(img, 10);
    cv::imshow("test1", img*255);
    img = clean(img);
    img = hbreak(img, 20);
    cv::imshow("test2", img*255);
    img = spur(img, 20);
    img = clean(img);
    cv::morphologyEx(img, img, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    img = thin(img, 10);


#ifdef DEBUG_CD_OPR_OUTPUT
    cv::imshow("Final CD", img*255);
    cv::waitKey(0);
#endif

    return img*255;
}