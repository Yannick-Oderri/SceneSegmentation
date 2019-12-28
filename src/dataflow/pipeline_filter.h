//
// Created by ynki9 on 12/27/19.
//

#ifndef PROJECT_EDGE_PIPELINE_FILTER_H
#define PROJECT_EDGE_PIPELINE_FILTER_H

#include <opencv2/opencv.hpp>
#include "queue_client.h"




template<class Data>
class PipeFilter {
protected:
    QueueClient<Data>* const in_queue_;
    QueueClient<Data>* const out_queue_;
    bool close_pipe_;

public:
    PipeFilter(QueueClient<Data>* const in_queue, QueueClient<Data>* const out_queue){
        this->in_queue_ = in_queue;
        this->out_queue_ = out_queue;
        this->close_pipe_ = false;
    }

    virtual void start();

};

template<class Data>
class CudaPipeFilter: protected PipeFilter<Data> {

public:
    CudaPipeFilter(QueueClient<Data>* const in_queue, QueueClient<Data>* const out_queue):
        PipeFilter<Data>(in_queue, out_queue){};

    virtual void start();
};



#endif //PROJECT_EDGE_PIPELINE_FILTER_H
