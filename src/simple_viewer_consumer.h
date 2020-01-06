//
// Created by ynki9 on 12/30/19.
//

#ifndef PROJECT_EDGE_SIMPLE_VIEWER_CONSUMER_H
#define PROJECT_EDGE_SIMPLE_VIEWER_CONSUMER_H

#include "pipeline_filter.h"
#include "libfreenect2/libfreenect2.hpp"
#include <opencv2/opencv.hpp>
#include <boost/log/trivial.hpp>

class SimpleViewerConsumer: ConsumerPipeFilter<libfreenect2::Frame*> {

public:
    SimpleViewerConsumer(QueueClient<libfreenect2::Frame*>* const in_queue ):
    ConsumerPipeFilter(in_queue){

    }

    void start(){
        int frame_count = 0;

        while(true){
            BOOST_LOG_TRIVIAL(info) << "Receiving Final Frame " << frame_count;

            in_queue_->waitData();
            libfreenect2::Frame* current_frame = in_queue_->front();

            cv::Mat cv_frame(current_frame->height, current_frame->width, CV_32F, current_frame->data);
            cv::imshow("Current Depth Frame", cv_frame);
            cv::waitKey(0);
            in_queue_->pop();

            frame_count++;
        }
    }
};


#endif //PROJECT_EDGE_SIMPLE_VIEWER_CONSUMER_H
