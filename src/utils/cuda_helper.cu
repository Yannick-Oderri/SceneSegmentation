//
// Created by ynki9 on 5/27/20.
//

__host__
void setupDepthMap(int width, int height, cv::Mat depth_map, cudaTextureObject_t& dev_depth_tex){
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
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&dev_depth_tex, &texRes, &texDescr, NULL));
}