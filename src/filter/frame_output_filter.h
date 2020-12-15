//
// Created by ynki9 on 12/15/20.
//

#ifndef PROJECT_EDGE_FRAME_OUTPUT_FILTER_H
#define PROJECT_EDGE_FRAME_OUTPUT_FILTER_H

#include "dataflow/pipeline_filter.h"
#include "dataflow/frame_observer.h"

class FrameOutputFilter: ConsumerPipeFilter<DepthFrameElement* const> {

public:
    FrameOutputFilter(QueueClient<DepthFrameElement* const>* const in_queue ):
    ConsumerPipeFilter(in_queue){

    }

    void start(){
        int frame_count = 0;

        while(true){
            BOOST_LOG_TRIVIAL(info) << "Receiving Final Frame " << frame_count;

            in_queue_->waitData();
            * current_frame = in_queue_->front();

            cv::Mat cv_frame(current_frame->height, current_frame->width, CV_32F, current_frame->data);
            cv::imshow("Current Depth Frame", cv_frame);
            cv::waitKey(0);
            in_queue_->pop();

            frame_count++;
        }
    }
};


#endif //PROJECT_EDGE_FRAME_OUTPUT_FILTER_H
