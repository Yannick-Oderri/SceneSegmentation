//
// Created by ynki9 on 12/30/19.
//

#ifndef PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
#define PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H

#include "dataflow/pipeline_filter.h"
#include <libfreenect2/libfreenect2.hpp>
#include <opencv2/opencv.hpp>
#include <boost/log/trivial.hpp>

class SimpleImageProducer: public ProducerPipeFilter<libfreenect2::Frame*> {
    libfreenect2::Frame* t_image_;
    cv::Mat image_;

public:
    SimpleImageProducer():
    ProducerPipeFilter(new QueueClient<libfreenect2::Frame*>()){}

    void initialize(){
        std::string file_path = "../../data/depth/test0.png";
        cv::Mat flip;
        flip = cv::imread(file_path, -1);

        if(flip.empty()) {
            BOOST_LOG_TRIVIAL(error) << "Image " << file_path << ": could not be loaded";
            return;
        }

        double min_val, max_val;
        cv::minMaxLoc(flip, &min_val, &max_val);
        // cv::flip(flip, image_, 0);
        flip.convertTo(image_, CV_32F, 1.0/max_val, 0);

        t_image_ = new libfreenect2::Frame(image_.rows, image_.cols, sizeof(float), image_.data);
    }

    void start(){
        int buffer_size = t_image_->width * t_image_->height * sizeof(float);
        int frame_count = 0;
        while(true){
            BOOST_LOG_TRIVIAL(info) << "Receiving Simple frame: " << frame_count;

            unsigned char* buffer = (unsigned char*)malloc(buffer_size);
            memcpy(buffer, image_.data, buffer_size);

            cv::Mat resize_mat(VIEWPORT_HEIGHT, VIEWPORT_WIDTH, CV_32F, buffer);
//            cv::imshow("simple_producer", resize_mat);
//            cv::waitKey(0);

            out_queue_->push(new libfreenect2::Frame(t_image_->height, t_image_->width, sizeof(float), buffer));
            frame_count++;

            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
};



#endif //PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
