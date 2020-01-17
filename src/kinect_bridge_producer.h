//
// Created by ynki9 on 12/29/19.
//

#ifndef PROJECT_EDGE_KINECT_BRIDGE_PRODUCER_H
#define PROJECT_EDGE_KINECT_BRIDGE_PRODUCER_H

/// [headers]
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>

#include "dataflow/pipeline_filter.h"

class FreenectPipeProducer: public ProducerPipeFilter<libfreenect2::Frame*> {
protected:
    libfreenect2::Freenect2 freenect2_;
    libfreenect2::Freenect2Device* freenect_dev_;
    libfreenect2::SyncMultiFrameListener freenect_listener_;
    libfreenect2::Registration* freenect_registration_;
    libfreenect2::FrameMap freenect_frames_;
    std::string serial_number_;
    int framecount_;
    bool enable_rgb_;
    bool enable_depth_;

public:

    /// Class Contructor
    FreenectPipeProducer():
    ProducerPipeFilter<libfreenect2::Frame*>(new QueueClient<libfreenect2::Frame*>()),
    freenect_listener_(libfreenect2::Frame::Depth | libfreenect2::Frame::Ir),
    freenect_dev_(nullptr),
    freenect_registration_(nullptr),
    freenect_frames_(),
    serial_number_(""),
    framecount_(0),
    enable_rgb_(true),
    enable_depth_(true){};

    void initialize();
    void initializeFreenectContext();
    void start();
    void shutdown();

    /// Pauses Freenect2 Stream
    inline void stopStream(){
        this->freenect_dev_->stop();
    }

    /// Closes Freenect2 Stream
    inline void closeStream(){
        this->freenect_dev_->close();
    }

    inline libfreenect2::Freenect2Device const* getFreenectDeviceID(){
        return this->freenect_dev_;
    }

    inline std::string getDeviceSerialNumber(){
        return this->serial_number_;
    }

    inline bool isInitialized(){
        return this->freenect_dev_ != nullptr;
    }
};


#endif //PROJECT_EDGE_KINECT_BRIDGE_PRODUCER_H
