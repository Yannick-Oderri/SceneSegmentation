//
// Created by ynki9 on 12/30/19.
//

#ifndef PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
#define PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H

#include "dataflow/pipeline_filter.h"
#include <libfreenect2/libfreenect2.hpp>
#include <opencv2/opencv.hpp>
#include <boost/log/trivial.hpp>
#include "frame.h"

class SimpleImageProducer: public ProducerPipeFilter<FrameElement* const> {
    libfreenect2::Frame* t_image_;
    cv::Mat image_;

    /** IR camera intrinsic calibration parameters.
   * Kinect v2 includes factory preset values for these parameters. They are used in depth image decoding, and Registration.
   */
    struct IrCameraParams
    {
        float fx; ///< Focal length x (pixel)
        float fy; ///< Focal length y (pixel)
        float cx; ///< Principal point x (pixel)
        float cy; ///< Principal point y (pixel)
        float k1; ///< Radial distortion coefficient, 1st-order
        float k2; ///< Radial distortion coefficient, 2nd-order
        float k3; ///< Radial distortion coefficient, 3rd-order
        float p1; ///< Tangential distortion coefficient
        float p2; ///< Tangential distortion coefficient
    };

public:
    SimpleImageProducer():
    ProducerPipeFilter(new QueueClient<FrameElement* const>()){}

    DepthCameraParams getDepthCameraParams(){
        DepthCameraParams camera_params;

        return camera_params;
    }

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
        const DepthCameraParams camera_params = this->getDepthCameraParams();
        while(this->close_pipe_){
            BOOST_LOG_TRIVIAL(info) << "Receiving Simple frame: " << frame_count;

            unsigned char* buffer = (unsigned char*)malloc(buffer_size);
            memcpy(buffer, image_.data, buffer_size);


            cv::Mat resize_mat(VIEWPORT_HEIGHT, VIEWPORT_WIDTH, CV_32F, buffer);

            DepthFrameElement* depth_content  = new DepthFrameElement(
                    t_image_->width,
                    t_image_->height,
                    sizeof(float),
                    reinterpret_cast<const float *const>(buffer),
                    &camera_params);

            out_queue_->push(new FrameElement(cv::Mat(), *depth_content));
            frame_count++;

            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        }
    }
};



#endif //PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
