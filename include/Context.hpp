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

using namespace cv;

struct Object {
	explicit Object(std::vector<Point>& pos, std::vector<float>& feature, int id = -1);

	std::vector<Point> pos;
	std::vector<float> feature;
	int id;
	std::vector<Point> trajectory;
	std::chrono::time_point<std::chrono::steady_clock> time;
};

struct BoundaryLine {
	explicit BoundaryLine(Point& p0, Point& p1);

	Point p0;
	Point p1;
	Scalar color;
	int lineThinkness;
	Scalar textColor;
	int textSize;
	int textThinkness;
	int count1;
	int count2;
};

struct Area {
	explicit Area(std::vector<Point>& contour);

	std::vector<Point> contour;
	size_t count;
	Scalar color;
};

inline Point vectorize(const Point& p1, const Point& p2);
inline bool onSegment(const Point& p1, const Point& p2, const Point& p3);
inline int direction(const Point& p1, const Point& p2, const Point& p3);
bool segmentsIntersect(const Point& p1, const Point& p2, const Point& p3, const Point& p4);

double computeAngle(const Point& p1, const Point& p2, const Point& p3, const Point& p4);