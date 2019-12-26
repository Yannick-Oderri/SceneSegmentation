//
// Created by ynki9 on 12/26/19.
//

#ifndef PROJECT_EDGE_PIPELINE_H
#define PROJECT_EDGE_PIPELINE_H

#include <string>
#include <stack>
#include <mutex>

#include "context.h"
#include "frame.h"

using PipelineID = unsigned int;

class IPipeline {
public:
    virtual void Execute();
};

class PipelineSource: IPipeline{

};

class PipelineFilter: PipelineSource{

};

/***************************************************************************//**
 *  Pipeline
 *  A Synhronous object use for processing incoming image information.
 ******************************************************************************/
class Pipeline: public IPipeline {
protected:
    std::stack<PipelineFilter> filters_;
    std::stack<ImageFrame> image_frame_buffer_;
    std::mutex buffer_mtx_;

    void processFrame();
    /// Fetches the next available frame.
    const ImageFrame &fetchFrame();


public:
    Pipeline(PipelineID id, AppContext* context);

    /// Synchronous Method for inserting frame element into processing pipeline.
    /// \param [in] frame Frame object to insert into pipeline for processing.
    void insertFrame(ImageFrame &frame);

};


#endif //PROJECT_EDGE_PIPELINE_H
