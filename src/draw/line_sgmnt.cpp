//
// Created by ynki9 on 2/21/20.
//

#include "line_sgmnt.h"

LineSegment::LineSegment(cv::Point start_pos, cv::Point end_pos):
        start_pos_(start_pos),
        end_pos_(end_pos),
        feature_concave_convex_(indeterminate),
        feature_depth_curve_(indeterminate),
        feature_left_right_(indeterminate){

}

LineSegment::LineSegment(Contour contour, std::pair<int, int>contour_region):
        start_pos_(contour[contour_region.first]),
        end_pos_(contour[contour_region.second]),
        feature_concave_convex_(indeterminate),
        feature_depth_curve_(indeterminate),
        feature_left_right_(indeterminate){

}

cv::Point LineSegment::getEndPos() {
    return this->end_pos_;
}

cv::Point LineSegment::getStartPos() {
    return this->start_pos_;
}


float LineSegment::getSlope() {
    return (end_pos_.y - start_pos_.y)/ (end_pos_.x - start_pos_.x);
}


float LineSegment::getLength() {
    return sqrt(pow(end_pos_.y - start_pos_.y, 2) + pow(end_pos_.x - start_pos_.x, 2));
}

tribool LineSegment::getConcavity() {
    return this->feature_concave_convex_;
}

tribool LineSegment::getDiscontinuity() {
    return this->feature_depth_curve_;
}

tribool LineSegment::getLocation() {
    return this->feature_left_right_;
}

void LineSegment::setDiscontinuity(bool val) {
    this->feature_depth_curve_ = val;
}
