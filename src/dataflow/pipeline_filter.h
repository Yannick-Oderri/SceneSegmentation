//
// Created by ynki9 on 12/27/19.
//

#ifndef PROJECT_EDGE_PIPELINE_FILTER_H
#define PROJECT_EDGE_PIPELINE_FILTER_H

#include <opencv2/opencv.hpp>
#include "queue_client.h"
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>

// #include "res/resource_mgr.h"

class ResMgr;

class AbstractPipeFilter {
public:
    virtual void start() = 0;
private:
};

template<class T_in, class T_out>
class PipeFilter: AbstractPipeFilter {
protected:
    QueueClient<T_in>* const in_queue_;
    QueueClient<T_out>* const out_queue_;
    bool close_pipe_;
    ResMgr* const res_mgr_;

public:
    PipeFilter(QueueClient<T_in>* const in_queue, QueueClient<T_out>* const out_queue):
            in_queue_(in_queue),
            out_queue_(out_queue),
            res_mgr_(nullptr){
        this->close_pipe_ = false;
    }
    PipeFilter(QueueClient<T_in>* const in_queue, QueueClient<T_out>* const out_queue, ResMgr* const res_mgr):
            in_queue_(in_queue),
            out_queue_(out_queue),
            res_mgr_(res_mgr){
        this->close_pipe_ = false;
    }

    inline QueueClient<T_out>* const getOutQueue() {
        return out_queue_;
    }

    inline QueueClient<T_in>* const getInQueue() {
        return in_queue_;
    }

    virtual void start() = 0;

};

template<class Data>
class ProducerPipeFilter: public PipeFilter<Data, Data> {
public:
    ProducerPipeFilter(QueueClient<Data>* const out_queue):
        PipeFilter<Data, Data>(nullptr, out_queue){};
    ProducerPipeFilter(QueueClient<Data>* const out_queue, ResMgr* const res_mgr):
        PipeFilter<Data, Data>(nullptr, out_queue, res_mgr){}

    virtual void start() = 0;
};

template<class Data>
class ConsumerPipeFilter: public PipeFilter<Data, Data> {
public:
    ConsumerPipeFilter(QueueClient<Data>* const in_queue):
            PipeFilter<Data, Data>(in_queue, nullptr){};

    virtual void start() = 0;
};

template<class T_in, class T_out>
class CudaPipeFilter: public PipeFilter<T_in, T_out> {
public:
    CudaPipeFilter(QueueClient<T_in>* const in_queue, QueueClient<T_out>* const out_queue):
        PipeFilter<T_in, T_out>(in_queue, out_queue){};

    virtual void start() = 0;
};

#endif //PROJECT_EDGE_PIPELINE_FILTER_H
