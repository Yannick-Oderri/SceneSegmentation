//
// Created by ynki9 on 2/6/20.
//

#include "contour_policy.h"
#include <tuple>
#include <utility>


using namespace std;
using LineSegment = vector<pair<cv::Point, cv::Point>>;


// Forward Declarations
vector<LineSegment>  lineSegmentExtraction(Contours contour_set, double tolerance);

void LineSegmentContourPolicy::executePolicy() {
    Contours contours = this->current_contour_data_->contours;

    // Segment Contours
    vector<LineSegment>segments =  lineSegmentExtraction(contours, 10.0f);

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


vector<LineSegment> lineSegmentExtraction(Contours contour_set, double tolerance) {
    vector<LineSegment> segment_list = vector<LineSegment>();

    for(Contour contour : contour_set) {
        LineSegment segments = LineSegment();
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
            cv::Point start_point = contour.at(first);
            cv::Point end_point = contour.at(last);
            segments.push_back(std::pair<cv::Point, cv::Point>(start_point, end_point));

            first = last;
            last = edge_count - 1;
        }

        segment_list.push_back(segments);
    }

    return segment_list;
}
