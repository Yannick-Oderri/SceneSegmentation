//
// Created by ynki9 on 5/25/20.
//

__global__
void contourDisc_kernel(int w, int h, cudaTextureObject_t depth_tex){
    for (int i = threadIdx.x; i < w; i += blockDim.x){

    }
}


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

__host__
ContourResult* launchCurveDiscOprKernel(cv::Mat& depth_map, int window_size){
    float4* contours;
    float4* dev_contours;
    ContourResult* dev_contour_results;

    int edge_count = 0;
    for(auto& segments: contour_segments){
    edge_count += segments.size();
    }
    contours = (float4*)malloc(sizeof(float4) * edge_count);

    int t_edge_index = 0;
    for(auto& segments: contour_segments){
    for(auto& line: segments){
    contours[t_edge_index] = make_float4(line.getStartPos().x, line.getStartPos().y, line.getEndPos().x, line.getEndPos().y);
    t_edge_index += 1;
    }
    }
    checkCudaErrors(cudaMalloc((void**)&dev_contours, sizeof(float4) * edge_count));
    checkCudaErrors(cudaMalloc((void**)&dev_contour_results, sizeof(ContourResult) * edge_count));
    checkCudaErrors(cudaMemcpy(dev_contours, contours, sizeof(float4) * edge_count, cudaMemcpyHostToDevice));

    cudaTextureObject_t depth_tex;
    setupDepthMap(depth_map.rows, depth_map.cols, depth_map, depth_tex);

    /// Execute Cuda kernel
    contourROIMean_kernel<<<1, edge_count>>>(dev_contours, dev_contour_results, depth_tex, window_size);

    ContourResult* contour_results = (ContourResult*)malloc(sizeof(ContourResult) * edge_count);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(contour_results, dev_contour_results, sizeof(ContourResult) * edge_count, cudaMemcpyDeviceToHost);

    /// cleanup temp contours
    free(contours);
    checkCudaErrors(cudaFree(dev_contour_results));
    checkCudaErrors(cudaFree(dev_contours));
    checkCudaErrors(cudaDestroyTextureObject(depth_tex));

    return contour_results;
}

extern "C"
cv::Mat* cuCurveDiscOperation(cv::Mat& depth_map, int window_size){

}