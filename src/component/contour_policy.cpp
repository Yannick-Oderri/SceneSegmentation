//
// Created by ynki9 on 2/6/20.
//

#include "contour_policy.h"
#include <tuple>
#include <utility>
#include <utils/helper.h>

#include "draw/line_sgmnt.h"
#include "draw/line_pair.h"

using namespace std;



// Forward Declarations
vector<vector<LineSegment>>  lineSegmentExtraction(Contours contour_set, double tolerance);
void calculateContourFeatures(vector<vector<LineSegment>> contour_segments, Contours contour_set, cv::Mat ddiscontinuity_map);
vector<LinePair> pairContourSegments(vector<vector<LineSegment>>& contour_segments, Contours& contour_set);
void drawLinePairs(vector<LinePair>& line_pairs, cv::Mat& color_image);

void LineSegmentContourPolicy::executePolicy() {
    Contours contours = this->current_contour_data_->contours;
    FrameElement frame_element = this->current_contour_data_->frame_element;

    // Segment Contours
    vector<vector<LineSegment>> contour_segments =  lineSegmentExtraction(contours, 3.0f);

    // Calculate contour features
    calculateContourFeatures(contour_segments, contours, frame_element.getDepthDiscontinuity());

    // Pair contours
    vector<LinePair> line_pairs = pairContourSegments(contour_segments, contours);
    cv::Mat drawing = cv::Mat::zeros( frame_element.getContourFrame().size(), CV_8UC3 );
    drawLinePairs(line_pairs, drawing);

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
            float len_ratio = abs((sgmnt_1.getLength() - sgmnt_2.getLength())/ (sgmnt_1.getLength() + sgmnt_2.getLength()));


            // Check if segments are parallel and equal length
            // abs(1 - abs(sgmnt_1.proj(sgmnt_2))) >= 1.0
            if( ang <= 0.07f && len_ratio < 0.3f)  { // perform pairing
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
 * Calculate line segment properties
 * @param contour_segments
 * @param contour_set
 * @param ddiscontinuity_map
 */
void calculateContourFeatures(vector<vector<LineSegment>> contour_segments, Contours contour_set, cv::Mat ddiscontinuity_map){
    for(int i = 0; i < contour_segments.size(); i++){
        auto contour = contour_set[i];
        auto segments = contour_segments[i];
        for(auto line_segment : segments){
            std::pair<int, int> contour_indecies = line_segment.getContourIndecies();
            int count = 0;
            for(int i = contour_indecies.first; i < contour_indecies.second; i++){
                // Check if segment is depth
                cv::Point point = contour[i];
                if(ddiscontinuity_map.at<uint8_t>(point) >= 200){
                    count++;
                }
            }
            /// Set discontinuity based on depth discontinuity map
            line_segment.setDiscontinuity(count >= (contour_indecies.second - contour_indecies.first) * 0.5);
        }
    }
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

void drawSegmentList(vector<vector<LineSegment>> contour_segments){
    cv::Mat image(480, 640, CV_8UC3);
    cv::RNG rng(3432764);
    for(auto segment : contour_segments){
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        for(auto line_segment: segment){
            cv::line(image, line_segment.getStartPos(), line_segment.getEndPos(), color);
        }
    }

    cv::imshow("Line Segments", image);
    cv::waitKey(0);
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
            if(line_segment.getLength() > 10)
                segments.push_back(line_segment);

            first = last;
            last = edge_count - 1;
        }

        contour_segments.push_back(segments);
    }
    drawSegmentList(contour_segments);
    return contour_segments;
}
