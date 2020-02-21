//
// Created by ynki9 on 12/29/19.
//

#include "kinect_bridge_producer.h"


#include <iostream>
#include <cstdlib>
#include <signal.h>
#include <thread>
#include <boost/log/trivial.hpp>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <GLFW/glfw3.h>


void FreenectPipeProducer::initializeFreenectContext() {
    libfreenect2::PacketPipeline *pipeline = 0;
    int types = 0;

/// [pipeline]
    pipeline = new libfreenect2::CpuPacketPipeline();
/// [pipeline]

/// [discovery]
    if(freenect2_.enumerateDevices() == 0) {
        BOOST_LOG_TRIVIAL(error) << "Kinect Device could not be found";
    }
    // Record device serial number
    serial_number_ = freenect2_.getDefaultDeviceSerialNumber();

    freenect_dev_ = freenect2_.openDevice(serial_number_, pipeline);
    if(freenect_dev_ == 0){
        BOOST_LOG_TRIVIAL(error) << "Failed to Open Device";
    }
/// [discovery]

    freenect_dev_->setColorFrameListener(&freenect_listener_);
    freenect_dev_->setIrAndDepthFrameListener(&freenect_listener_);

    BOOST_LOG_TRIVIAL(info) << "device serial: " << freenect_dev_->getSerialNumber();
    BOOST_LOG_TRIVIAL(info) << "device firmware: " << freenect_dev_->getFirmwareVersion();
}

void FreenectPipeProducer::start() {
    if(!freenect_dev_->startStreams(enable_rgb_, enable_depth_)){
        BOOST_LOG_TRIVIAL(error) << "Failed to start freenect stream";
        return;
    }

/// [registration setup]
    freenect_registration_ = new libfreenect2::Registration(freenect_dev_->getIrCameraParams(),
            freenect_dev_->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4);
    libfreenect2::Frame registered(512, 424, 4);
/// [registration setup]
    double current_time = 0;
    double previous_time = 0;
    bool write_file = false;
    while(this->close_pipe_ == false){
        current_time = glfwGetTime();

        if (!freenect_listener_.waitForNewFrame(freenect_frames_, 10*1000)){
            BOOST_LOG_TRIVIAL(warning) << "Kinect Device Timeout";
        }

        BOOST_LOG_TRIVIAL(info) << "Receiving Kinect V2 Frame:" << framecount_ << " " << current_time - previous_time;
        libfreenect2::Frame *rgb = freenect_frames_[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = freenect_frames_[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = freenect_frames_[libfreenect2::Frame::Depth];

        DepthFrameElement dframe (
                depth->width,
                depth->height,
                sizeof(float),
                depth->data,
                freenect_dev_->getIrCameraParams());


        out_queue_->push(dframe);

        this->framecount_++;
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        freenect_listener_.release(freenect_frames_);
        previous_time = current_time;
    }

    // End Freenect driver streaming
    this->freenect_dev_->stop();
}

void FreenectPipeProducer::initialize() {
    this->initializeFreenectContext();
}