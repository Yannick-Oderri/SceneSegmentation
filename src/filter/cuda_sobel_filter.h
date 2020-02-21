//
// Created by ynki9 on 12/30/19.
//

#ifndef PROJECT_EDGE_CUDA_SOBEL_FILTER_H
#define PROJECT_EDGE_CUDA_SOBEL_FILTER_H

#include <cuda_runtime.h>
#include <libfreenect2/frame_listener.hpp>

#include "context/context.h"
#include "dataflow/pipeline_filter.h"

class CudaSobelFilter: public PipeFilter<libfreenect2::Frame*, libfreenect2::Frame*>{
private:
    CudaDevice cuda_dev_id_;
    libfreenect2::Frame* working_frame_;
    cudaResourceDesc cuda_in_res_desc_;
    cudaTextureDesc cuda_device_in_tex_desc_;
    cudaTextureObject_t cuda_device_in_tex_;
    float* cuda_device_in_buffer_;
    float* cuda_device_out_buffer_;
    size_t cuda_frame_pitch_;
    float sobel_param_fscale_;
    int frame_width_;
    int frame_height_;

public:
    /**
     * Pipeline Object Constructor
     * @param in_pipe
     */
    CudaSobelFilter(QueueClient<libfreenect2::Frame*>* in_pipe):
            PipeFilter<libfreenect2::Frame*, libfreenect2::Frame*>(in_pipe, new QueueClient<libfreenect2::Frame*>()),
            frame_width_(640),
            frame_height_(480),
            sobel_param_fscale_(1.0f){}
    /**
     * Register CUDA objects
     * @param cuda_device
     */
    __global__ void intialize(CudaDevice cuda_device);

    void start();
    __global__ void cuda_start();
};


#endif //PROJECT_EDGE_CUDA_SOBEL_FILTER_H
