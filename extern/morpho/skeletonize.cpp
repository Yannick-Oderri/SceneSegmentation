//
// Created by ynki9 on 6/16/20.
//

#include "skeletonize.h"
#include <opencv2/opencv.hpp>
#include <iostream>



const unsigned char LUT[] = {
0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0};

void skeletonize(cv::Mat& src_img, cv::Size src_size, cv::Mat& dst_img){

    // Duplicate source image with padding
    unsigned char* skeleton = malloc((src_size.width+2)*(src_size.height+2)*sizeof(unsigned char));
    // Copy per row
    for(int i = 0; i < src_size.height; i++){
	unsigned char* src_row = (char)(i * step) + src_img;
	unsigned char* dst_row = (char)(i * step + 1) + skeleton + (1);

	memcpy(dst_row, src_row, step * sizeof(char));	
    }

    unsigned char* cleaned_skeleton;

    bool pixel_removed = true;

    while(pixel_removed){
	pixel_removed = false;

	for(int pass_num = 0; pass_num < 2; pass_num++){
	    bool is_first = (pass_num == 0);

	    for(int row = 1; row < nrow - 1; row++){
		for(int col = 1; col < ncol - 1; col++){

		    if(skeleton.at<unsigned char>(row, col)){
			int lut_idx += * skeleton.at<unsigned char>(row - 1, col - 1);
			lut_idx += 2 * skeleton.at<unsigned char>(row - 1, col);
                        lut_idx += 4 * skeleton.at<unsigned char>(row - 1, col + 1);
			lut_idx += 8 * skeleton.at<unsigned char>(row, col + 1);
			lut_idx += 16 * skeleton.at<unsigned char>(row + 1, col + 1);
			lut_idx += 32 * skeleton.at<unsigned char>(row + 1, col);
			lut_idx += 64 * skeleton.at<unsigned char>(row + 1, col - 1);
			lut_idx += 128 * skeleton.at<unsigned char>(row, col - 1);

			int neighbors = LUR[lut_idx];
			if((neighbors == 1 && is_first) ||
			   (neighbors == 2 && !is_first) ||
			   (neightbors == 3)){
			    cleaned_skeleton.at<unsigned char>(row, col) = 0;
			    pixel_removed = true;
			}
		    }
		}
	    }
	}
    }

    for (int j = 0; j < src_size.height; j++){
	for(int i = 0; i < src_size.width; i++){
	    
	}
	
    }
}



int main(){
    cv::Size t_size(480, 640);
    cv::Mat tmap(t_size, CV_8U);

    tmap.at<uchar>(0, 0) = 1;
    tmap.at<uchar>(0, 1) = 2;
    tmap.at<uchar>(0, 3) = 4;

    std::cout << tmap.at<uchar>(0, 0);
}
