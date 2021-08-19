#include "implementHarris.h"

int main(int argc, char** argv) {

	Mat image;
	vector< pair<int, int> > cornerCoordinates;
	gradientTypes derType = gradientTypes::Sobel;

	harrisFilterParams hp;
	implementHarris imHr;

	image = imread("maps/m1_edit.png", 1);
	cout << "image cols: " << image.cols << endl;
	cout << "image rows: " << image.rows << endl;
	auto start = high_resolution_clock::now();

	// Few edge cases for different parameters of the algorithm where the input parameters are out of range.
	if (hp.gaussSigma <= 0.0) {
		cout << "Error: Standard deviation of the Gaussian filter must be greater than zero." << endl;
		return 0;
	}
	if (hp.thresHoldNMS >= 1.0 || hp.thresHoldNMS <= 0.0) {
		cout << "Error: Non-maximum suppression threshold fraction must be between 0 and 1." << endl;
		return 0;
	}
	if (hp.gradSumWindowSize % 2 == 0 || hp.gradSumWindowSize <= 1 || hp.gradSumWindowSize >= ((image.rows <= image.cols) ? image.rows : image.cols)) {
		cout << "Error: Grad sum window size must be an odd and positive number and must be smaller than the smaller dimension of the input image." << endl;
		return 0;
	}
	if (hp.nmsWindowSize <= 0 || hp.nmsWindowSize >= ((image.rows <= image.cols) ? image.rows : image.cols)) {
		cout << "Error: Non-maximum suppression window size must be greater than zero and must be smaller than the smaller dimension of the input image." << endl;
		return 0;
	}
	if (hp.nmsWindowSeparation < 1 || hp.nmsWindowSeparation > hp.nmsWindowSize) {
		cout << "Error: Non-maximum suppression window separation must be greater than zero and less than or equal to non-maximum suppression window size." << endl;
		return 0;
	}

	/*
	 * Call the primary function of the implementHarris class.
	 * This function returns the detected corner coordinates
	 * of the given color or grayscale image.
	 */
	cornerCoordinates = imHr.harrisCornerDetect(image, &hp, derType);

	/*
	 * Set points in the corresponding corner pixels of the output image.
	 */
	for (auto const& points : cornerCoordinates)
		circle(image, Point(points.second, points.first), 3, Scalar(0, 255, 0), 1);

	imwrite("out.jpg", image);

	auto stop = high_resolution_clock::now();

	auto elapsedTime = duration<double, std::milli>(stop - start).count();

	std::cout << "Total Execution time: " << elapsedTime << " milliseconds" << endl;

	return 0;
}