//
// Created by ynki9 on 2/5/20.
//

#include "contour_processor.h"

void ContourProcessorPipeFilter::initialize() {

}


void ContourProcessorPipeFilter::start() {
    while(true){
        getInQueue()->waitData();
        ContourAttributes* contour_data = getInQueue()->front();

        this->exec_policy->setContourData(contour_data);


        this->exec_policy->executePolicy();

        delete contour_data;

        getInQueue()->pop();
        
    }
}

