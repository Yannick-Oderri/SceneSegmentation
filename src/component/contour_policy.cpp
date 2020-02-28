//
// Created by ynki9 on 2/6/20.
//

#include <algorithm>
#include "contour_policy.h"
#include <tuple>
#include <utility>
#include <utils/helper.h>

#include "draw/line_sgmnt.h"
#include "draw/line_pair.h"

using namespace std;



// Forward Declarations
vector<vector<LineSegment>>  lineSegmentExtraction(Contours contour_set, double tolerance);
void calculateContourFeatures(vector<vector<LineSegment>>& contour_segments, Contours contour_set, FrameElement&);
vector<LinePair> pairContourSegments(vector<vector<LineSegment>>& contour_segments, Contours& contour_set);
void drawLinePairs(vector<LinePair>& line_pairs, cv::Mat& color_image);
void drawSegmentList(vector<vector<LineSegment>>& contour_segments, int mode = 1);
std::vector<cv::Point> generateWindowCooridnates(std::pair<cv::Point, cv::Point> line, int window_size, int buffer_size);
std::pair<int, int> determineROIMeans(cv::Mat depth_img, vector<cv::Point> roi, cv::Mat& contour_mask, double& contour_overlap_p, double& contour_overlap_n);
double determineROIDepthDiscMeans(cv::Mat depth_img, vector<cv::Point> roi);


void LineSegmentContourPolicy::executePolicy() {
    Contours contours = this->current_contour_data_->contours;
    FrameElement frame_element = this->current_contour_data_->frame_element;

    // Segment Contours
    vector<vector<LineSegment>> contour_segments =  lineSegmentExtraction(contours, 3.0f);

    // Calculate contour features
    calculateContourFeatures(contour_segments, contours, frame_element);
    // render line feature
    for(int i = 1; i < 5; i++){
        drawSegmentList(contour_segments, i);
    }
    cv::waitKey(0);

    // Pair contours
    vector<LinePair> line_pairs = pairContourSegments(contour_segments, contours);
    cv::Mat drawing = cv::Mat::zeros( frame_element.getContourFrame().size(), CV_8UC3 );
    drawLinePairs(line_pairs, drawing);

}

void mergeLines(vector<LineSegment> segments, FrameElement& frame_element){

}

/**
 * Calculate line segment properties
 * @param contour_segments
 * @param contour_set
 * @param ddiscontinuity_map
 */
void calculateContourFeatures(vector<vector<LineSegment>>& contour_segments, Contours contour_set, FrameElement& frame_element){
    cv::Mat dd_img = frame_element.getDepthDiscontinuity();
    cv::Mat cd_img = frame_element.getCurveDiscontinuity();

    for(int i = 0; i < contour_segments.size(); i++){
        auto& contour = contour_set[i];
        auto& segments = contour_segments[i];
        cv::Mat depth_img = frame_element.getDepthFrameData()->getcvMat();
        // generate contour mask
        cv::Mat contour_mask(frame_element.getDepthDiscontinuity().rows, frame_element.getDepthDiscontinuity().cols, CV_32F);
        cv::drawContours(contour_mask, contour_set, i, cv::Scalar(1),  -1);

        // perform calculation on contour segments
        for(auto& line_segment : segments){
            // Determine Region of interest averages
            vector<cv::Point> roi_polies = generateWindowCooridnates(line_segment.asPointPair(), 5, 0);
            double countp, countn;
            std::pair<int, int> roi_means = determineROIMeans(depth_img, roi_polies, contour_mask, countp, countn);

            std::pair<int, int> contour_indecies = line_segment.getContourIndecies();
            vector<int> line_depth_vals(contour_indecies.second - contour_indecies.first);
            int line_depth_sum = 0;
            for(int i = 0; i < contour_indecies.second - contour_indecies.first; i++){
                // Check if segment is depth
                cv::Point point = contour[contour_indecies.first + i];
                int depth_val = (int)depth_img.at<float>(point);
                line_depth_sum += depth_val;
                line_depth_vals.at(i) = depth_val;
            }

            roi_polies = generateWindowCooridnates(line_segment.asPointPair(), 2, 0);
            int line_depth_mean = std::count_if(line_depth_vals.begin(), line_depth_vals.end(), [](int i){return i != 0;});
            if(line_depth_mean != 0)
                line_depth_mean = line_depth_sum / line_depth_mean;

            /// Set discontinuity based on depth discontinuity map
            double depth_disc_mean = determineROIDepthDiscMeans(dd_img, roi_polies);
            if(depth_disc_mean > 0.1f) {
                line_segment.setDiscontinuity(true);

                /// set edge pose base on roi depths and amount of overlap with contour mask
                if ((roi_means.first <= roi_means.second) && (countn >= countp))
                    line_segment.setPose(true);
                else if((roi_means.second <= roi_means.first) && (countp >= countn))
                    line_segment.setPose(false);

            }else{ /// for curve discontinuity base
                line_segment.setDiscontinuity(false);

                /// set convexity if average line depth is less than or equal roi average
                if ((line_depth_mean <= roi_means.first) && (line_depth_mean <= roi_means.second))
                    line_segment.setConvexity(true);
                else
                    line_segment.setConvexity(false);
            }

        }
    }
}

void drawLinePairs(vector<LinePair>& line_pairs, cv::Mat& color_image){
    cv::RNG rng(3432764);

    for(auto line_pair: line_pairs){
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        LineSegment& sgmnt_1 = line_pair[0];
        LineSegment& sgmnt_2 = line_pair[1];
        cv::line(color_image, sgmnt_1.getStartPos(), sgmnt_1.getEndPos(), color);
        cv::line(color_image, sgmnt_2.getStartPos(), sgmnt_2.getEndPos(), color);
    }

    cv::imshow("Paired Lines", color_image);
    cv::waitKey(0);
}

vector<LinePair> pairContourSegments(vector<vector<LineSegment>>& contour_segments, Contours& contour_set){
    vector<LinePair> line_pairs;
    for(int i = 0; i < contour_segments.size(); i++){
        vector<LineSegment>& segments = contour_segments[i];

        Combinations cs(segments.size(), 2);
        vector<int> used_indecies;
        while(!cs.completed){
            Combinations::combination_t c = cs.next();
            LineSegment& sgmnt_1 = segments[c[0]];
            LineSegment& sgmnt_2 = segments[c[1]];
            // If lines has been used already skip pairing
            if(std::count(used_indecies.begin(), used_indecies.end(), c[0]) != 0||
               std::count(used_indecies.begin(), used_indecies.end(), c[1]) != 0){
                continue;
            }

            float dot = sgmnt_1.dot(sgmnt_2);
            float ang = sgmnt_1.proj(sgmnt_2) / sgmnt_2.getLength();

            // length ration
            float len_ratio = 1 - abs((sgmnt_1.getLength() - sgmnt_2.getLength())/ (sgmnt_1.getLength() + sgmnt_2.getLength()));


            // Check if segments are parallel and equal length
            // abs(1 - abs(sgmnt_1.proj(sgmnt_2))) >= 1.0
            if( ang <= 0.07f &&
                len_ratio > 0.6f &&
                sgmnt_1.getPose() != sgmnt_2.getPose() &&
                sgmnt_1.isConcave() == false &&
                sgmnt_2.isConcave() == false)  { // perform pairing
                line_pairs.push_back(LinePair(sgmnt_1, sgmnt_2));
                // Keep track of lines that have been paired already [TODO This method could be imporved]
                used_indecies.push_back(c[0]);
                used_indecies.push_back(c[1]);
            }
        }
    }

    return line_pairs;
}


/**
 * Segments
        (See Jain, Rangachar and Schunck, "Machine Vision", McGraw-Hill
        1996. pp 194-196)
 * @param first
 * @param last
 * @param contour
 * @return
 */
pair<int, float> maxSegmentDeviation(int first, int last, Contour contour){
    vector<double> deviation_vals(last - first);

    // line spliting algorithm
    // x*(y1-y2) + y*(x2-x1) + y2*x1 - y1*x2 = 0
    cv::Point first_point = contour[first];
    cv::Point last_point = contour[last];

    double center_line_len = sqrt(pow(last_point.x - first_point.x, 2) + pow(last_point.y - first_point.y, 2));

    double y1my2 = first_point.y - last_point.y;
    double x2mx1 = last_point.x - first_point.x;
    double param_b = last_point.y*first_point.x - first_point.y*last_point.x;

    double max_dist = 0;
    int max_dist_index;
    if(center_line_len > 0.0001) {
        for (int i = first; i < last; i++) {
            cv::Point t_point = contour[i];
            double dist = abs(t_point.x * y1my2 + t_point.y * x2mx1 + param_b) / center_line_len;
            if (max_dist != max(dist, max_dist)) {
                max_dist = max(dist, max_dist);
                max_dist_index = i;
            }
        }
    }else{ // If distance between start and end points is 0, find distance of all points with first point
        for (int i = first; i < last; i++) {
            cv::Point t_point = contour[i];
            double dist = sqrt(pow(t_point.x - first_point.x, 2) + pow(t_point.y - first_point.y, 2));
            if (max_dist != max(dist, max_dist)) {
                max_dist = max(dist, max_dist);
                max_dist_index = i;
            }
        }
    }

    return pair<int, double>(max_dist_index, max_dist);
}

void drawSegmentList(vector<vector<LineSegment>>& contour_segments, int mode){
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::RNG rng(3432764);
    cv::Scalar color1;
    cv::Scalar color2;
    string name;

    if(mode == 2){ // Right left render mode
        color1 = cv::Scalar(0, 255, 0);
        color2 = cv::Scalar(0, 0, 255);
        name = "pose";
    }else if(mode == 3){ // convex concave render mode
        color1 = cv::Scalar(255, 0, 255); // magenta
        color2 = cv::Scalar(255, 0, 0);
        name = "convexity";
    }else if(mode == 4){ // depth curve discontinuity
        color1 = cv::Scalar(0, 255, 255);
        color2 = cv::Scalar(255, 255, 0);
        name = "discontinuity";
    }

    for(auto& segment : contour_segments){
        if(mode == 1) // random color as default
            color1 = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        for(auto& line_segment: segment){
            if((line_segment.isPoseRight() && mode == 2) ||
               (line_segment.isConvex() && mode == 3) ||
               (line_segment.isDepthDiscontinuity() && mode == 4) ||
               mode == 1){
                cv::line(image, line_segment.getStartPos(), line_segment.getEndPos(), color1);
            }else if((line_segment.isPoseLeft() && mode == 2) ||
                    (line_segment.isConcave() && mode == 3) ||
                    (line_segment.isCurveDiscontinuity() && mode == 4)){
                cv::line(image, line_segment.getStartPos(), line_segment.getEndPos(), color2);
            }
        }
    }

    cv::imshow("Segment Feature: " + name, image);
}

vector<vector<LineSegment>> lineSegmentExtraction(Contours contour_set, double tolerance) {
    vector<vector<LineSegment>> contour_segments = vector<vector<LineSegment>>();

    for(Contour contour : contour_set) {
        vector<LineSegment> segments;
        int edge_count = contour.size();
        int first = 0;
        int last = edge_count - 1;

        // Search for
        while(first < last){
            pair<int, double> deviation = maxSegmentDeviation(first, last, contour);
            while(deviation.second > tolerance){
                last = deviation.first;
                deviation = maxSegmentDeviation(first, last, contour);
            }
            LineSegment line_segment(contour, std::pair<int, int>(first, last));
            if(line_segment.getLength() > 1)
                segments.push_back(line_segment);

            first = last;
            last = edge_count - 1;
        }

        contour_segments.push_back(segments);
    }
    // drawSegmentList(contour_segments);
    return contour_segments;
}

/**
 * Determine window/roi around line polygon-coordinates
 * @param line
 * @param window_size
 * @param buffer_size
 * @return
 */
std::vector<cv::Point> generateWindowCooridnates(std::pair<cv::Point, cv::Point> line, int window_size, int buffer_size){
//    std::pair<cv::Point, cv::Point> line(
//            cv::Point(t_line.first.y, t_line.first.x),
//            cv::Point(t_line.second.y, t_line.second.y));

    float dy = abs(line.second.y - line.first.y);
    float dx = abs(line.second.x - line.first.x);
    std::pair<cv::Point, cv::Point> res;
    cv::Point pt1, pt2, pt3, pt4;

    // Determine line orientation
    if(dy <= dx){
        pt1 = cv::Point(line.first.x - buffer_size, line.first.y - window_size - buffer_size);
        pt2 = cv::Point(line.first.x + buffer_size, line.first.y + window_size + buffer_size);
        pt3 = cv::Point(line.second.x - buffer_size, line.second.y - window_size - buffer_size);
        pt4 = cv::Point(line.second.x + buffer_size, line.second.y + window_size + buffer_size);
    }else{
        pt1 = cv::Point(line.first.x - window_size - buffer_size, line.first.y - buffer_size);
        pt2 = cv::Point(line.first.x + window_size + buffer_size, line.first.y + buffer_size);
        pt3 = cv::Point(line.second.x - window_size - buffer_size, line.second.y - buffer_size);
        pt4 = cv::Point(line.second.x + window_size + buffer_size, line.second.y + buffer_size);
    }

    cv::Point temp_1 = (((pt1 + pt3) / 2) - ((pt2 + pt4) / 2));
    float mag_1 = sqrtf(pow(temp_1.x, 2) + pow(temp_1.y, 2));

    cv::Point temp_2 = (((pt1 + pt4) / 2) - ((pt2 + pt3) / 2));
    float mag_2 = sqrtf(pow(temp_2.x, 2) + pow(temp_2.y, 2));

    vector<cv::Point> win(8);
    if (mag_1 > mag_2){ // vertical oriented line
        // win_p
        win.at(0) = line.first;
        win.at(1) = line.second;
        win.at(2) = pt4;
        win.at(3) = pt2;

        // win_n
        win.at(4) = pt1;
        win.at(5) = pt3;
        win.at(6) = line.second;
        win.at(7) = line.first;
    }else{ // horizontal oriented line
        // win_p
        win.at(0) = line.first;
        win.at(1) = pt4;
        win.at(2) = line.second;
        win.at(3) = pt2;

        // win_n
        win.at(4) = pt1;
        win.at(5) = line.second;
        win.at(6) = pt3;
        win.at(7) = line.first;
    }

    return win;
}

std::pair<int, int> determineROIMeans(cv::Mat depth_img, vector<cv::Point> roi, cv::Mat& contour_mask, double& contour_overlap_p, double& contour_overlap_n){
    std::pair<int, int> res;

    vector<vector<cv::Point>> t_contour(2); // define temporary contour for drawing rois
    t_contour.at(0) = vector<cv::Point>(roi.begin(), roi.end() - 4);
    t_contour.at(1) = vector<cv::Point>(roi.begin() + 4, roi.end());

    for(int i = 0; i < 2; i++) {
        cv::Mat mask = cv::Mat::zeros(depth_img.rows, depth_img.cols, CV_32F);

        // plot roi
        cv::drawContours(mask, t_contour, i, cv::Scalar(1), -1);

        // determine count
        int count = cv::countNonZero(mask);
        double mask_sum = depth_img.dot(mask);
        if(i == 0) {
            res.first = mask_sum / count;
            contour_overlap_p = contour_mask.dot(mask);
        }else {
            res.second = mask_sum / count;
            contour_overlap_n = contour_mask.dot(mask);
        }
    }

    return res;

}


double determineROIDepthDiscMeans(cv::Mat depth_img, vector<cv::Point> roi){
    std::pair<int, int> res;
    cv::Mat mask = cv::Mat::zeros(depth_img.rows, depth_img.cols, CV_8U);

    vector<vector<cv::Point>> t_contour(2); // define temporary contour for drawing rois
    t_contour.at(0) = vector<cv::Point>(roi.begin(), roi.end() - 4);
    t_contour.at(1) = vector<cv::Point>(roi.begin() + 4, roi.end());

    for(int i = 0; i < 2; i++) {
        // plot roi
        cv::drawContours(mask, t_contour, i, cv::Scalar(1), -1);
    }

    // determine count
    int count = cv::contourArea(t_contour[0]) * 2;
    double mask_sum = depth_img.dot(mask) / 255;
    return mask_sum/count;

}