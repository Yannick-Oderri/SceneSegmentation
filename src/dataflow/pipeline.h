//
// Created by ynki9 on 12/26/19.
//

#ifndef PROJECT_EDGE_PIPELINE_H
#define PROJECT_EDGE_PIPELINE_H

#include <string>
#include <stack>
#include <mutex>

#include "context.h"
#include "pipeline_filter.h"

using PipelineID = unsigned int;

enum PipeFilterType{
    CudaSobelFilter = 0,
};

class IPipeline {
public:
    virtual void Execute();
};


/***************************************************************************//**
 *  Pipeline
 *  A Synhronous object use for processing incoming image information.
 ******************************************************************************/
class Pipeline: public IPipeline {
protected:

public:
    Pipeline(PipelineID id, AppContext* context);


    void Execute() override;
    void AppendFilter(PipeFilterType filter_type);
};


#endif //PROJECT_EDGE_PIPELINE_H
