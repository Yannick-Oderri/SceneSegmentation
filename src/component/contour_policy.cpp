//
// Created by ynki9 on 2/6/20.
//

#include "contour_policy.h"
#include <tuple>
#include <utility>


using namespace std;
using LineSegment = vector<pair<cv::Point, cv::Point>>;

void LineSegmentContourPolicy::executePolicy() {
    Contours contours = this->current_contour_data_->contours;


}


void lineSegmentExtraction(Contours contour_set, double tolerance) {
    for(Contour contour : contour_set) {
        int edge_count = contour.size();
        vector<LineSegment> segment_list = vector<LineSegment>();
        int first = 0;
        int last = edge_count - 1;

        // Search for
        while(first < last){
            pair<int, double> deviation = maxSegmentDeviation(first, last, contour);
            while(deviation.second > tolerance){
                last = deviation.first;
                deviation = maxSegmentDeviation(first, last, contour);
            }
        }

    }
}

pair<int, float> maxSegmentDeviation(int first, int last, Contour contour){
    vector<double> deviation_vals(first-last);

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
    for(int i = first; i < last; i++){
        cv::Point t_point = contour[i];
        double dist = (t_point.x * y1my2 + t_point.y * x2mx1 + param_b) / center_line_len;
        if(max_dist != max(dist, max_dist)){
            max_dist = max(dist, max_dist);
            max_dist_index = i;
        }
    }

    return pair<int, double>(max_dist_index, max_dist);
}