#pragma once

#define _USE_MATH_DEFINES
#define PRECISION 1000.0

#include <vector>
#include <chrono>
#include <math.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


struct Object {
	explicit Object(std::vector<cv::Point>& pos, std::vector<float>& feature, int id = -1);

	std::vector<cv::Point> pos;
	std::vector<float> feature;
	int id;
	std::vector<cv::Point> trajectory;
	std::chrono::time_point<std::chrono::steady_clock> time;
};

struct BoundaryLine {
	explicit BoundaryLine(cv::Point& p0, cv::Point& p1);

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

inline cv::Point vectorize(const cv::Point& p1, const cv::Point& p2);
inline bool onSegment(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3);
inline int direction(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3);
bool segmentsIntersect(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, const cv::Point& p4);

double computeAngle(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, const cv::Point& p4);