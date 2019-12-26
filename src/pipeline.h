//
// Created by ynki9 on 12/26/19.
//

#ifndef PROJECT_EDGE_PIPELINE_H
#define PROJECT_EDGE_PIPELINE_H

#include <string>
#include <stack>

#include "context.h"

class IPipeline {
public:
    virtual void Execute();
};

class PipelineSource: IPipeline{

};

class PipelineFilter: PipelineSource{

};

class Pipeline: public IPipeline {
protected:
    std::stack<PipelineFilter> filters;

public:
    Pipeline(std::string name, AppContext* context);
};


#endif //PROJECT_EDGE_PIPELINE_H
