//
// Created by ynki9 on 7/25/20.
//

#include "k4a_bridge_producer.h"

void k4aImageProducer::initialize() {
    // K4A Configuration
    m_k4a_config_ = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    m_k4a_config_.camera_fps = K4A_FRAMES_PER_SECOND_30;
    m_k4a_config_.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    m_k4a_config_.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    m_k4a_config_.color_resolution = K4A_COLOR_RESOLUTION_720P;

    // This means that we'll only get captures that have both color and
    // depth images, so we don't need to check if the capture contains
    // a particular type of image.
    //
    m_k4a_config_.synchronized_images_only = true;

    this->m_k4a_device_ = k4a::device::open(K4A_DEVICE_DEFAULT);
    this->m_k4a_device_.start_cameras(&m_k4a_config_);

    if(!m_k4a_device_.get_capture(&m_k4a_capture_, std::chrono::milliseconds(1000))) {
        BOOST_LOG_TRIVIAL(error) << "[Streaming Service] Runtime error: k4a_device_get_capture() failed";
    }

    // Setup device calibration
    k4a_device_get_calibration(m_k4a_device_.handle(), m_k4a_config_.depth_mode,
            m_k4a_config_.color_resolution, &m_k4a_calibration_);

    // Setup Transform for color to depth
    m_k4a_transform_ = k4a_transformation_create(&m_k4a_calibration_);
}

FrameElement* k4aImageProducer::pollCurrentFrame(int timeout){
    FrameElement* frame = nullptr;
    if(m_k4a_device_.get_capture(&m_k4a_capture_, std::chrono::milliseconds(timeout))) {
        const k4a::image k4a_depth_image = m_k4a_capture_.get_depth_image(); // Azure Kinect Depth Image
        const k4a::image k4a_color_image_t = m_k4a_capture_.get_color_image(); // Azure Kinect Color Image

        // Color Image Transformation
        k4a_image_t transformed_color_image_h = NULL;
        k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                     k4a_depth_image.get_width_pixels(),
                                                     k4a_depth_image.get_height_pixels(),
                                                     k4a_depth_image.get_width_pixels() * 4 * (int)sizeof(uint8_t),
                                                     &transformed_color_image_h);
        k4a::image k4a_color_image(transformed_color_image_h);
        if (K4A_RESULT_SUCCEEDED != k4a_transformation_color_image_to_depth_camera(m_k4a_transform_,
                                                                                   k4a_depth_image.handle(),
                                                                                   k4a_color_image_t.handle(),
                                                                                   k4a_color_image.handle())){
            printf("Failed ot compute tranformed image \n");
            return nullptr;
        }

        m_frame_count_ += 1;



        // Convert color imag eto opencv Mat
        unsigned char* color_buffer = (unsigned char *)(malloc(k4a_color_image.get_size()));
        memcpy(color_buffer, k4a_color_image.get_buffer(), k4a_color_image.get_size());
        cv::Mat color_img(
                k4a_color_image.get_height_pixels(),
                k4a_color_image.get_width_pixels(),
                CV_8UC4, color_buffer, k4a_color_image.get_stride_bytes());

        // Convert depth image to opencv Mat
        unsigned char* depth_buffer = (unsigned char *)malloc(k4a_depth_image.get_size());
        memcpy(depth_buffer, k4a_depth_image.get_buffer(), k4a_depth_image.get_size());

        cv::Mat t_depth_img(
                k4a_depth_image.get_height_pixels(),
                k4a_depth_image.get_width_pixels(),
                CV_16U, depth_buffer, k4a_depth_image.get_stride_bytes());
//        cv::normalize(t_depth_img, t_depth_img, 0, 65000, CV_MINMAX, CV_16U);
//        cv::imshow("depth image", t_depth_img);
//        cv::waitKey(0);

        // Normalize Depth Image ?? This could be done on GPU
        cv::Mat t_mask = (t_depth_img < 650) | (t_depth_img > 1700);
        t_depth_img = t_depth_img.clone();
        t_depth_img.setTo(0, t_mask);
        //t_depth_img = processFrame(t_depth_img); // Averages frame

        cv::Mat depth_image;
        cv::Mat ndepth_image;
        t_depth_img.convertTo(depth_image, CV_32F, 1.0, 0);
        cv::normalize(t_depth_img, ndepth_image, 0.0, 1.0, CV_MINMAX, CV_32F);

        float* depth_buffer1 = (float*)malloc(depth_image.step*depth_image.cols);
        memcpy(depth_buffer1, depth_image.ptr(), depth_image.step*depth_image.cols);
        float* ndepth_buffer = (float*)malloc(ndepth_image.step*ndepth_image.cols);
        memcpy(ndepth_buffer, ndepth_image.ptr(), ndepth_image.step*ndepth_image.cols);



        DepthFrameElement* depth_content = new DepthFrameElement(
                k4a_depth_image.get_width_pixels(),
                k4a_depth_image.get_height_pixels(),
                sizeof(float),
                (float *) depth_buffer1,
                (float *) ndepth_buffer,
                (unsigned char*)depth_buffer);

        long timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(k4a_depth_image.get_system_timestamp()).count();
        frame = new FrameElement(m_frame_count_, color_img, depth_content, timestamp);

        k4a_image_release(transformed_color_image_h);

    }

    return frame;
}

void k4aImageProducer::start(){
    int frame_count = 0;
    while(!this->m_end_stream_){
        FrameElement* frame = pollCurrentFrame(60);
        if(frame == nullptr){
            std::cout << "Invalid Frame Received: " << std::endl;
            continue;
        }

        out_queue_->push(frame);

        m_frame_count_++;
        std::this_thread::sleep_for(std::chrono::milliseconds(m_frame_delay_));
    }
}

cv::Mat k4aImageProducer::processFrame(cv::Mat & new_frame) {
    m_frame_queue_.push_front(new_frame);
    if(m_frame_queue_.size() > m_frame_queue_count_)
        m_frame_queue_.pop_back();

    cv::Mat t_frame = cv::Mat::zeros(new_frame.rows, new_frame.cols, new_frame.type());
    cv::Mat t_ones = cv::Mat::ones(new_frame.rows, new_frame.cols, new_frame.type());
    cv::Mat t_frame2 = cv::Mat::zeros(new_frame.rows, new_frame.cols, new_frame.type());
    for(int i=0; i < m_frame_queue_.size(); i++){
        t_frame = t_frame + m_frame_queue_[i];
        cv::Mat tf =  m_frame_queue_[i] > 1;
        cv::add(t_frame2, t_ones, t_frame2, tf, CV_16U);
    }
    t_frame = t_frame / t_frame2;

    return t_frame;
}
