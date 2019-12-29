//
// Created by ynki9 on 12/27/19.
//

#ifndef PROJECT_EDGE_PIPELINE_FILTER_H
#define PROJECT_EDGE_PIPELINE_FILTER_H

#include <opencv2/opencv.hpp>
#include "queue_client.h"

class AbstractPipeFilter {
public:
    virtual void start();
private:
};

template<class Data>
class PipeFilter: AbstractPipeFilter {
protected:
    QueueClient<Data>* const in_queue_;
    QueueClient<Data>* const out_queue_;
    bool close_pipe_;

public:
    PipeFilter(QueueClient<Data>* const in_queue, QueueClient<Data>* const out_queue):
            in_queue_(in_queue), out_queue_(out_queue){
        this->close_pipe_ = false;
    }

    inline QueueClient<Data>* getOutQueue() const{
        return out_queue_;
    }

    inline QueueClient<Data>* getInQueue() const{
        return in_queue_;
    }

    virtual void start() = 0;

};

template<class Data>
class ProducerPipeFilter: protected PipeFilter<Data> {
public:
    ProducerPipeFilter(QueueClient<Data>* const out_queue):
        PipeFilter<Data>(nullptr, out_queue){};

    virtual void start() = 0;
};

template<class Data>
class ConsumerPipeFilter: protected PipeFilter<Data> {
public:
    ConsumerPipeFilter(QueueClient<Data>* const in_queue):
            PipeFilter<Data>(in_queue, nullptr){};

    virtual void start() = 0;
};

template<class Data>
class CudaPipeFilter: protected PipeFilter<Data> {
public:
    CudaPipeFilter(QueueClient<Data>* const in_queue, QueueClient<Data>* const out_queue):
        PipeFilter<Data>(in_queue, out_queue){};

    virtual void start() = 0;
};

#endif //PROJECT_EDGE_PIPELINE_FILTER_H
