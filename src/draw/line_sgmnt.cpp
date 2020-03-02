//
// Created by ynki9 on 2/21/20.
//

#include "line_sgmnt.h"

LineSegment::LineSegment(cv::Point start_pos, cv::Point end_pos):
        start_pos_(start_pos),
        end_pos_(end_pos),
        feature_convex_concave_(indeterminate),
        feature_depth_curve_(indeterminate),
        feature_right_left_(indeterminate){

}

LineSegment::LineSegment(Contour contour, std::pair<int, int>contour_region):
        contour_(&contour),
        contour_region_(contour_region),
        start_pos_(contour[contour_region.first]),
        end_pos_(contour[contour_region.second]),
        feature_convex_concave_(indeterminate),
        feature_depth_curve_(indeterminate),
        feature_right_left_(indeterminate){

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

tribool LineSegment::getConvexity() {
    return this->feature_convex_concave_;
}

tribool LineSegment::getDiscontinuity() {
    return this->feature_depth_curve_;
}

tribool LineSegment::getPose() {
    return this->feature_right_left_;
}

void LineSegment::setDiscontinuity(bool val) {
    this->feature_depth_curve_ = val;
}

float LineSegment::getAngle() {
    float res = atan(-this->getSlope());
    return (res < 0) ? res + 180 : res;
}

float LineSegment::dot(LineSegment &rhs) {
    cv::Point l_vec = cv::Point(end_pos_.x - start_pos_.x, end_pos_.y - start_pos_.y);
    cv::Point r_vec = cv::Point(rhs.end_pos_.x - rhs.start_pos_.x, rhs.end_pos_.y - rhs.start_pos_.y);
    return (float)((l_vec.x * r_vec.x) + (l_vec.y * r_vec.y));
}

float LineSegment::proj(LineSegment &rhs) {
    float dot = this->dot(rhs);
    return dot / this->getLength();
}

std::pair<int, int> LineSegment::getContourIndecies() {
    return this->contour_region_;
}

std::pair<cv::Point, cv::Point> LineSegment::asPointPair() {
    return std::pair<cv::Point, cv::Point>(this->start_pos_, this->end_pos_);
}

void LineSegment::setPose(bool val) {
    this->feature_right_left_ = val;
}

void LineSegment::setConvexity(bool val) {
    this->feature_convex_concave_ = val;
}

bool LineSegment::isCurveDiscontinuity() {
    return this->feature_depth_curve_ == false;
}

bool LineSegment::isDepthDiscontinuity() {
    return this->feature_depth_curve_ == true;
}

bool LineSegment::isPoseLeft() {
    return this->feature_right_left_ == false;
}

bool LineSegment::isPoseRight() {
    return this->feature_right_left_ == true;
}

bool LineSegment::isConcave() {
    return this->feature_convex_concave_ == false;
}

bool LineSegment::isConvex() {
    return this->feature_convex_concave_ == true;
}

void LineSegment::setLinePlacement(bool b) {
    this->feature_background_foreground_ = b;
}

bool LineSegment::isBackground() {
    return this->feature_background_foreground_ == true;
}