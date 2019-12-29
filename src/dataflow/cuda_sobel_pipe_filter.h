//
// Created by ynki9 on 12/27/19.
//

#ifndef PROJECT_EDGE_CUDASOBELPIPEFILTER_H
#define PROJECT_EDGE_CUDASOBELPIPEFILTER_H

#include "pipeline_filter.h"

#define CUDA_SOBEL "cuda_sobel"

class CudaSobelPipeFilter;

class CudaSobelPipeFilterFactory: protected AbstractPipelineFilterFactory {
public:
    CudaSobelPipeFilterFactory() {
        AbstractPipeFilter::registerType(CUDA_SOBEL, this);
    }
    virtual AbstractPipeFilter* create() {
        return new CudaSobelPipeFilter();
    }
}
static CudaSobelPipeFilterFactory global_CudaSobelPipeFilterFactory;

class CudaSobelPipeFilter: protected CudaPipeFilter<cv::Mat> {
public:
    CudaSobelPipeFilter(QueueClient<cv::Mat>* const in_queue, QueueClient<cv::Mat>* const out_queue):
            CudaPipeFilter<cv::Mat>(in_queue, out_queue){};

    void start();
};
#endif //PROJECT_EDGE_CUDASOBELPIPEFILTER_H
