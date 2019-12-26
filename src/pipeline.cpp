//
// Created by ynki9 on 12/26/19.
//

#include "pipeline.h"

void Pipeline::insertFrame(ImageFrame &frame) {
    // Ensure Data is thread safe.
    std::lock_guard<std::mutex> guard(this->buffer_mtx_);

    // add image to frame pipeline frame buffer.
    this->image_frame_buffer_.push(frame);

    return;
}

void Pipeline::processFrame() {
    const ImageFrame img = this->fetchFrame();

}

const ImageFrame &Pipeline::fetchFrame() {
    std::lock_guard<std::mutex> guard(this->buffer_mtx_);
    const ImageFrame &target_frame = this->image_frame_buffer_.top();

    return target_frame;
}