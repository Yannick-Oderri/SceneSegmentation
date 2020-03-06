//
// Created by ynki9 on 3/6/20.
//

#include "gl_depth_img_filter.h"


void GLDepthImageFilter::initialize() {

}

void GLDepthImageFilter::start() {
    while(true){
        getInQueue()->waitData();
        FrameElement* const frame_element = getInQueue()->front();

        this->exec_policy->setFrameData(frame_element);
        this->exec_policy->executePolicy();

    }
}