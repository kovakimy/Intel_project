#pragma once

#define _USE_MATH_DEFINES
#define PRECISION 1000.0

#include <vector>
#include <chrono>
#include <math.h>
#include <iostream>
#include <random>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking/tracking_by_matching.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

struct BoundaryLine {
	explicit BoundaryLine(const cv::Point& p0, const cv::Point& p1);

	cv::Point p0;
	cv::Point p1;
	cv::Scalar color;
	int lineThinkness;
	cv::Scalar textColor;
	int textSize;
	int textThinkness;
	int count1;
	int count2;
};

struct Area {
	explicit Area(std::vector<cv::Point>& contour);

	std::vector<cv::Point> contour;
	size_t count;
	cv::Scalar color;
};

struct Parameters {
	size_t mode;
	cv::Mat frame;
	std::string windowName;
	std::vector<cv::Point> areaContour;
	std::vector<cv::Point> linePoints;
};

inline cv::Point vectorize(const cv::Point& p1, const cv::Point& p2);
inline bool onSegment(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3);
inline int direction(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3);
bool segmentsIntersect(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, const cv::Point& p4);
double computeAngle(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, const cv::Point& p4);
void callback(int event, int x, int y, int flag, void* userdata);
std::vector<cv::Scalar> generateColors(size_t size);