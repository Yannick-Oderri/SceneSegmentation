
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions


extern 'C'
void cu_determineROIMean(cv::Mat depth_img, vector<vector<cv::Point>> contours, cv::Mat& contour_mask, double& contour_overlap_p, double& contour_overlap_n) {
/// Prepare contour
}