#include "cuda_sobel_pipe_filter.h"

void CudaSobelPipeFilter::start() {
    cv::Mat mat;
    while (this->close_pipe_ == false) {
        in_queue_->waitData(); // wait for data at input buffer
        mat = in_queue_->front();
        
    }
}