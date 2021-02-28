#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <opencv2/core/types.hpp>
#include "Context.hpp"

#pragma once

using namespace std;
using namespace cv;

/*
class Object
{
private:
	Point p1;
	Point p2;
	double timer = 0;
	vector<Point> trajectoty;
public:
	Object(Point a, Point b);
	void set_points(Point p1_, Point p2_);
	Point get_first_point() const;
	Point get_sec_point() const;
	void setTimer(int time);
	int getTimer() const;
	void add_to_trajectory(cv::Point p);
};
*/



class ObjectTracker {
private:
	double E_t;
	double E_s;
	int next_id = 0;
	int max_not_found_time = 5;
	double similarityThreshold = 0.4;
	vector<Object> current_objects;
	vector<Object> prev_objects;
	void Predict();
public:
	ObjectTracker(double not_found_segment_cost, double not_found_object_cost);
	vector<int> SetStartObjects(vector<Object> objects_centers);
	vector<int> Track(vector<Object>& segments);
};