//
// Created by ynk on 2/24/20.
//

#ifndef PROJECT_EDGE_LINE_PAIR_H
#define PROJECT_EDGE_LINE_PAIR_H

#include "draw/line_sgmnt.h"
#include "draw/drawable.h"

class LinePair: public Drawable{
private:
    std::pair<LineSegment&, LineSegment&> line_segments_;
    cv::Scalar foreground_color_;
    cv::Scalar background_color_;

public:
    /**
     * Constructor
     * @param sgmnt_1
     * @param sgmnt_2
     */
    LinePair(LineSegment& sgmnt_1, LineSegment& sgmnt_2):
            line_segments_(sgmnt_1, sgmnt_2){}

    /**
     * Line pair subscriptable opeator 0, 1, throws error for > 1
     * @param index
     * @return
     */
    LineSegment& operator[](int index);


    cv::Scalar getForegroundColor() override;
    cv::Scalar getBackgroundColor() override;
    void draw2D(DrawableFrameBuffer& framebuffer) override;
    void draw3D() override ;
};



#endif //PROJECT_EDGE_LINE_PAIR_H
