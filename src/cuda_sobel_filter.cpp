//
// Created by ynki9 on 12/30/19.
//

#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <driver_types.h>
#include <boost/log/trivial.hpp>
#include "cuda_sobel_filter.h"
#include "cuda_sobel_filter_kernel.h"

__global__ void CudaSobelFilter::intialize(CudaDevice cuda_device) {
    int buffer_size = frame_width_ * frame_height_ * sizeof(float);
    /// Allocate memory for Framebuffer to store temporary image data
    unsigned char* buffer = (unsigned char*) std::malloc(buffer_size);
    working_frame_ = new libfreenect2::Frame(frame_width_, frame_height_, sizeof(float), buffer);

    /// Allocate necessary memory elements on cuda device
    cudaMallocPitch(&cuda_device_in_buffer_, &this->cuda_frame_pitch_,
            frame_width_*sizeof(float), frame_height_);
    cudaMallocPitch(&cuda_device_out_buffer_, &this->cuda_frame_pitch_,
            frame_width_*sizeof(float), frame_height_);


    /// Resource Descriptor describes data to texture
    memset(&cuda_in_res_desc_, 0x0, sizeof(cuda_in_res_desc_));
    cuda_in_res_desc_.resType = cudaResourceTypePitch2D;
    cuda_in_res_desc_.res.pitch2D.devPtr = cuda_device_in_buffer_;
    cuda_in_res_desc_.res.pitch2D.width = frame_width_;
    cuda_in_res_desc_.res.pitch2D.height = frame_height_;
    cuda_in_res_desc_.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    cuda_in_res_desc_.res.pitch2D.pitchInBytes = cuda_frame_pitch_;

    /// Texture Descriptor describes how data should be textured
    memset(&cuda_device_in_tex_desc_, 0x0, sizeof(cuda_device_in_tex_desc_));
    cuda_device_in_tex_desc_.normalizedCoords = false;
    cuda_device_in_tex_desc_.filterMode       = cudaFilterModePoint;
    cuda_device_in_tex_desc_.addressMode[0]   = cudaAddressModeWrap;
    cuda_device_in_tex_desc_.readMode = cudaReadModeElementType;

    // cudaCreateTextureObject(&cuda_device_in_tex_, &cuda_in_res_desc_, &cuda_device_in_tex_desc_, NULL);
}

void CudaSobelFilter::start() {
    this->cuda_start();
}

__global__ void CudaSobelFilter::cuda_start(){
    int buffer_size = frame_width_ * frame_height_ * sizeof(float);
    int frame_count = 0;

    while(frame_count < 10){
        this->in_queue_->waitData();
        libfreenect2::Frame* incomming_frame = in_queue_->front();
//        std::memcpy(working_frame_->data, incomming_frame->data, buffer_size);

        BOOST_LOG_TRIVIAL(info) << "Receiving Sobel Frame " << frame_count;

        /// Copy image to cuda device
        auto result = cudaMemcpy2D(cuda_device_in_buffer_, cuda_frame_pitch_,
                     incomming_frame->data, frame_width_*sizeof(float), frame_width_* sizeof(float),
                     frame_height_, cudaMemcpyHostToDevice);
        if(result != cudaSuccess){
            BOOST_LOG_TRIVIAL(error) << "Error while trying to allocate cuda device memory";
        };

        cudaCreateTextureObject(&cuda_device_in_tex_, &cuda_in_res_desc_, &cuda_device_in_tex_desc_, NULL);

        sobelFilter(cuda_device_in_buffer_, cuda_device_out_buffer_,
                frame_width_, frame_height_, sobel_param_fscale_, cuda_device_in_tex_);

        unsigned char* host_out_data = (unsigned char*)malloc(buffer_size);

//        /// Run in sync with cuda threads
        cudaDeviceSynchronize();

        cudaMemcpy2D(host_out_data, frame_width_* sizeof(float),
                     cuda_device_out_buffer_, cuda_frame_pitch_, frame_width_* sizeof(float),
                     frame_height_, cudaMemcpyDeviceToHost);

        libfreenect2::Frame* out_frame = new libfreenect2::Frame(frame_width_, frame_height_, sizeof(float), host_out_data);
        /// Remove image from top of previous queue
        in_queue_->pop();
        out_queue_->push(out_frame);

        frame_count++;

    }
}