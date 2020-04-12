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
    tribool feature_right_left_;
    tribool feature_convex_concave_;
    tribool feature_depth_curve_;
    tribool feature_background_foreground_;

    cv::Point2f start_pos_;
    cv::Point2f end_pos_;
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
    cv::Point2f getStartPos() const;

    /**
     * Get line segment end pos
     * @return
     */
    cv::Point2f getEndPos() const ;


    /**
     * get slope of line segment
     * @return
     */
    float getSlope() const;

    /**
     * get length of line segment
     * @return
     */
    float getLength() const;

    /**
     * Get Convexity of Line segment
     * @return
     */
    tribool getConvexity() const;

    /**
     * Is line segment convex
     * @return
     */
    bool isConvex() const;

    /**
     * Is line segment concave
     * @return
     */
    bool isConcave() const;

    /**
     * Get locatoin of line segment
     * @return
     */
    tribool getPose() const;

    /**
     * Is Line position right of contour
     * @return
     */
    bool isPoseRight();

    /**
     * Is Line segment position left of contour
     * @return
     */
    bool isPoseLeft();

    /**
     * Get discontinuity of line segment
     * @return
     */
    tribool getDiscontinuity() const;

    /**
     * Is line segment depth discontinuity
     * @return
     */
    bool isDepthDiscontinuity();

    /**
     * Is line segmnet curve discontinuity
     * @return
     */
    bool isCurveDiscontinuity();

    /**
     * Sets line to depth or curve discontinuity
     * @param val true for depth false for curve
     */
    void setDiscontinuity(bool val);

    /**
     * Return indicies of start and end points held in segmnets contour object
     * @return
     */
    std::pair<int, int> getContourIndecies();

    /**
     * Get angle of line segment with horizontal
     * @return
     */
    float getAngle() const;

    /**
     * Returns dot product between 2 lines
     * @param rhs
     * @return
     */
    float dot(LineSegment& rhs);

    /**
     * Returns projection between 2 line segments
     * @param rhs
     * @return
     */
    float proj(LineSegment& rhs);

    /**
     * Returns pair of cv::Points representing line start and end positions
     * @return
     */
    std::pair<cv::Point2f, cv::Point2f> asPointPair();

    /**
     * Set right left line pose
     * @param b true == right false == left
     */
    void setPose(bool b);

    /**
     * Set line convexity
     * @param b true == convex false == concave
     */
    void setConvexity(bool b);

    void setLinePlacement(bool b);

    bool isBackground();

    /**
     * Returns normalized orientation of line segment
     * @return
     */
    cv::Point2f getOrientation();
};


#endif //PROJECT_EDGE_LINE_SGMNT_H
