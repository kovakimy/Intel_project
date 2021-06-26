#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <opencv2/core/types.hpp>
#include "../include/Context.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/tracking/tracking_legacy.hpp>

#pragma once


class ObjectTracker {
private:
	bool is_first = false;
	float E_t;
	float E_s;
	int next_id = 0;
	int max_not_found_time = 5;
	float similarityThreshold = 0.55;// The best value will be calculated later
	std::vector<Object> current_objects;
	std::vector<Object> prev_objects;
	//cv::Ptr<cv::legacy::MultiTracker> trackers;
	//std::vector<cv::Ptr<cv::legacy::Tracker> > algorithms;
	//void Predict();
public:
	std::vector<Object> getCurrentObjects()
	{
		return current_objects;
	}
	ObjectTracker(float not_found_segment_cost, float not_found_object_cost);
	//vector<int> SetStartObjects(vector<Object> objects_centers);
	std::vector<Object> Track(std::vector<Object>& segments, cv::Mat& frame);
};