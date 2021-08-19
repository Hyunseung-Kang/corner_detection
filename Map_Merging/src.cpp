#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define threshold	100
#define kernel_size	3

bool detect_corner(Mat input_img);


int main(void) {
	Mat map1 = imread("map1_test.jpg", 0);
	
	for(int i=int(kernel_size/2); i<map1.cols-(int(kernel_size/2)); i++)
		for (int j = int(kernel_size/2); j < map1.rows - (int(kernel_size/2)); j++) {
			Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_8UC1);
			for(int kernel_i = 0; kernel_i<kernel_size; kernel_i++)
				for (int kernel_j = 0; kernel_j < kernel_size; kernel_j++) {
					kernel.at<uchar>(kernel_j, kernel_i) = map1.at<uchar>(j-(int(kernel_size/2)) + kernel_j, i-(int(kernel_size/2)) + kernel_i);
				}
			if (detect_corner(kernel)) {
				map1.at<uchar>(j, i) = 255;
			}
		}
	imshow("map1_after", map1);
	waitKey(0);
	return 0;
}

bool detect_corner(Mat input_img) {
	int k = input_img.at < uchar>(0, 0);
	if ((abs(input_img.at<uchar>(0, 0) - input_img.at<uchar>(0, input_img.cols-1)) > threshold) && (abs(input_img.at<uchar>(0, 0) - input_img.at<uchar>(0, input_img.cols - 1)) < 255))
		if ((abs(input_img.at<uchar>(0, 0) - input_img.at<uchar>(input_img.rows-1, 0)) > threshold)
			|| (abs(input_img.at<uchar>(0, input_img.cols-1) - input_img.at<uchar>(input_img.rows-1, input_img.cols-1)) > threshold))
			return true;
		else
			return false;

	else if ((abs(input_img.at<uchar>(input_img.rows-1, 0) - input_img.at<uchar>(input_img.rows-1, input_img.cols-1)) > threshold) && (abs(input_img.at<uchar>(input_img.rows - 1, 0) - input_img.at<uchar>(input_img.rows - 1, input_img.cols - 1)) <255))
		if ((abs(input_img.at<uchar>(0, 0) - input_img.at<uchar>(input_img.rows-1, 0)) > threshold)
			|| (abs(input_img.at<uchar>(0, input_img.cols-1) - input_img.at<uchar>(input_img.rows-1, input_img.cols-1)) > threshold))
			return true;
		else
			return false;
	else
		return false;
}