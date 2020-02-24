//
// Created by ynki9 on 2/21/20.
//

#ifndef PROJECT_EDGE_LINE_SGMNT_H
#define PROJECT_EDGE_LINE_SGMNT_H

#include <boost/logic/tribool.hpp>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace boost::logic;
using Contour = std::vector<cv::Point>;

class LineSegment {
private:
    tribool feature_left_right_;
    tribool feature_concave_convex_;
    tribool feature_depth_curve_;

    cv::Point start_pos_;
    cv::Point end_pos_;
    Contour* contour_;
    std::pair<int, int> contour_region_;


public:
    LineSegment(cv::Point start_pos, cv::Point end_pos);

    LineSegment(Contour, std::pair<int, int> contour_region_);

    LineSegment() = delete;

    /**
     * Get line segment start pos
     * @return
     */
    cv::Point getStartPos();

    /**
     * Get line segment end pos
     * @return
     */
    cv::Point getEndPos();


    /**
     * get slope of line segment
     * @return
     */
    inline float getSlope();

    /**
     * get length of line segment
     * @return
     */
    inline float getLength();

    /**
     * Get Concavity of Line segment
     * @return
     */
    inline tribool getConcavity();

    /**
     * Get locatoin of line segment
     * @return
     */
    inline tribool getLocation();

    /**
     * Get discontinuity of line segment
     * @return
     */
    inline tribool getDiscontinuity();
    inline void setDiscontinuity(bool val);

    inline std::pair<int, int> getContourIndecies();

};


#endif //PROJECT_EDGE_LINE_SGMNT_H
