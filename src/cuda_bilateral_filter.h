//
// Created by ynki9 on 12/30/19.
//

#ifndef PROJECT_EDGE_CUDA_BILATERAL_FILTER_H
#define PROJECT_EDGE_CUDA_BILATERAL_FILTER_H

#include <cuda_runtime.h>

#include "context.h"
#include "dataflow/pipeline_filter.h"
#include <libfreenect2/libfreenect2.hpp>

class CudaBilateralFilter: public PipeFilter<libfreenect2::Frame*, libfreenect2::Frame*>{
private:
public:
    CudaBilateralFilter(QueueClient<libfreenect2::Frame*> in_queue):
            PipeFilter(in_queue, new QueueClient<libfreenect2::Frame*>()){}

    __global__ void initialize(CudaDevice device_id);
    __global__ void cudaStart();
    void start();

};


#endif //PROJECT_EDGE_CUDA_BILATERAL_FILTER_H
