//
// Created by ynki9 on 2/6/20.
//

#ifndef PROJECT_EDGE_CONTOUR_POLICY_H
#define PROJECT_EDGE_CONTOUR_POLICY_H

#include "component/pipeline_policy.h"


/**
 * Abstract policy for processing contour data
 */
class ContourPolicy: public PipelinePolicy{
public:
    /**
     * Constructor
     */
    ContourPolicy(){}
    virtual void setContourData(ContourAttributes*) = 0;
    virtual bool executePolicy() = 0;
};

/**
 *
 */
class LineSegmentContourPolicy: public ContourPolicy{
private:
    ContourAttributes * current_contour_data_;

public:
    LineSegmentContourPolicy():
            ContourPolicy(){}

    void setContourData(ContourAttributes * contour_data){
        this->current_contour_data_ = contour_data;
    }
    bool executePolicy();
};

#endif //PROJECT_EDGE_CONTOUR_POLICY_H
