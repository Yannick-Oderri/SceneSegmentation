//
// Created by ynki9 on 2/5/20.
//

#include "contour_processor.h"
#include <GLFW/glfw3.h>
#include <boost/log/trivial.hpp>

void ContourProcessorPipeFilter::initialize() {

}


void ContourProcessorPipeFilter::start() {
    while(true){
        getInQueue()->waitData();
        ContourAttributes* contour_data = getInQueue()->front();

        this->exec_policy->setContourData(contour_data);


        this->exec_policy->executePolicy();
        
        double frame_time = contour_data->frame_element.getFrameTime();
        double elapse_time = glfwGetTime() - frame_time;
        BOOST_LOG_TRIVIAL(info) << "Total Frame Time: " << elapse_time;
        
        // temporary delete data
        delete contour_data;

        getInQueue()->pop();
        
    }
}

