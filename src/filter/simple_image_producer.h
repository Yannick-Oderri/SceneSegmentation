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

class SimpleImageProducer: public ProducerPipeFilter<FrameElement* > {
    libfreenect2::Frame* t_image_;
    cv::Mat image_;
    cv::Mat color_image_;

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
    ProducerPipeFilter(new QueueClient<FrameElement* >()){}

    DepthCameraParams getDepthCameraParams(){
        DepthCameraParams camera_params;

        camera_params.fx = 550;
        camera_params.fy = 550;
        camera_params.cx = 640 / 2;
        camera_params.cy = 480 / 2;

        return camera_params;
    }

    void initialize(){
        std::string file_path = "../data/depth/test0.png";
        std::string color_img_file_path = "../data/depth/ctest0.png";
        cv::Mat img;
        img = cv::imread(file_path, -1);

        color_image_ = cv::imread(color_img_file_path);

        if(img.empty() || color_image_.empty()) {
            BOOST_LOG_TRIVIAL(error) << "Image " << file_path << ": could not be loaded";
            return;
        }

        img.convertTo(image_, CV_32F, 1.0, 0);

        t_image_ = new libfreenect2::Frame(image_.cols, image_.rows, sizeof(float), image_.ptr());
    }

    void start(){
        int buffer_size = t_image_->width * t_image_->height * sizeof(float);
        int frame_count = 0;
        const DepthCameraParams camera_params = this->getDepthCameraParams();
        while(this->close_pipe_ == false){
            BOOST_LOG_TRIVIAL(info) << "Receiving Simple frame: " << frame_count;

            unsigned char* buffer = (unsigned char*)malloc(buffer_size);
            memcpy(buffer, image_.data, buffer_size);


            cv::Mat resize_mat(VIEWPORT_HEIGHT, VIEWPORT_WIDTH, CV_32F, buffer);

            DepthFrameElement* depth_content  = new DepthFrameElement(
                    t_image_->width,
                    t_image_->height,
                    sizeof(float),
                    (float*)buffer,
                    &camera_params);

            // Generate Color Image mat
            cv::Mat t_col;
            color_image_.copyTo(t_col);

            out_queue_->push(new FrameElement(t_col, *depth_content));
            frame_count++;

            std::this_thread::sleep_for(std::chrono::milliseconds(200000));
        }
    }

    /**
     * End Pipeline loop
     */
    void signalEnd(){
        this->close_pipe_ = true;
    }

};



#endif //PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
