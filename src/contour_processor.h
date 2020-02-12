//
// Created by ynki9 on 2/5/20.
//

#ifndef PROJECT_EDGE_CONTOUR_PROCESSOR_H
#define PROJECT_EDGE_CONTOUR_PROCESSOR_H

#include "component/pipeline_policy.h"
#include "pipeline_filter.h"
#include "frame.h"

class ContourProcessorPipeFilter: public PipeFilter<ContourAttributes* , RenderableElement*> {

private:
    ContourPolicy* const exec_policy;

public:
    ContourProcessorPipeFilter(QueueClient<ContourAttributes*>* in_queue, ContourPolicy* const processing_policy):
            PipeFilter(in_queue, new QueueClient<RenderableElement* >()),
            exec_policy(processing_policy){

    }

    /**
     * Initialize Pipeline filter
     */
    void initialize();

    /**
     * Execute Pipeline filter thread. Functoin loops until application ends
     */
    void start();
};


#endif //PROJECT_EDGE_CONTOUR_PROCESSOR_H
