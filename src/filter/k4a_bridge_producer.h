//
// Created by ynki9 on 7/25/20.
//

#ifndef PROJECT_EDGE_K4A_BRIDGE_PRODUCER_H
#define PROJECT_EDGE_K4A_BRIDGE_PRODUCER_H

#include <opencv2/opencv.hpp>
#include <boost/log/trivial.hpp>
#include <thread>
#include "dataflow/pipeline_filter.h"
#include "res/resource_mgr.h"
#include "frame.h"
#include <k4a/k4a.hpp>

class k4aImageProducer: public ProducerPipeFilter<FrameElement*> {
public:
    k4aImageProducer(ResMgr* const res_mgr, int frame_delay=30, int frame_queue_len=10):
        ProducerPipeFilter(new QueueClient<FrameElement* >(), res_mgr),
        m_end_stream_(false), m_frame_delay_(frame_delay), m_frame_queue_count_(frame_queue_len){}
    void initialize();

    void start();

    FrameElement* pollCurrentFrame(int timeout);


    /**
     * Signal End Streaming
     */
    void signalEnd();

private:
    cv::Mat processFrame(cv::Mat&);

    k4a::device m_k4a_device_;
    k4a_device_configuration_t m_k4a_config_;
    bool m_end_stream_;
    const int m_frame_delay_;
    int m_frame_count_;
    k4a::capture m_k4a_capture_;
    const int m_frame_queue_count_;
    std::deque<cv::Mat> m_frame_queue_;
};


#endif //PROJECT_EDGE_K4A_BRIDGE_PRODUCER_H
