//
// Created by ynki9 on 2/6/20.
//

#include "contour_policy.h"
#include <tuple>
#include <utility>

#include "draw/line_sgmnt.h"

using namespace std;


// Forward Declarations
vector<vector<LineSegment>>  lineSegmentExtraction(Contours contour_set, double tolerance);

void LineSegmentContourPolicy::executePolicy() {
    Contours contours = this->current_contour_data_->contours;
    FrameElement frame_element = this->current_contour_data_->frame_element;

    // Segment Contours
    vector<vector<LineSegment>>contour_segments =  lineSegmentExtraction(contours, 3.0f);

    // Calculate contour features


    //contour_segments
}

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
                if(ddiscontinuity_map.at<int8_t >(point) >= 200){
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
            cv::Point start_point = contour.at(first);
            cv::Point end_point = contour.at(last);
            LineSegment line_segment(contour, std::pair<int, int>(first, last));
            segments.push_back(line_segment);

            first = last;
            last = edge_count - 1;
        }

        contour_segments.push_back(segments);
    }
    drawSegmentList(contour_segments);
    return contour_segments;
}
