#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <opencv2/core/types.hpp>
#include "../include/Context.hpp"

#pragma once


class ObjectTracker {
private:
	float E_t;
	float E_s;
	int next_id = 0;
	int max_not_found_time = 5;
	float similarityThreshold = 0.75;// The best value will be calculated later
	std::vector<Object> current_objects;
	std::vector<Object> prev_objects;
	//void Predict();
public:
	ObjectTracker(float not_found_segment_cost, float not_found_object_cost);
	//vector<int> SetStartObjects(vector<Object> objects_centers);
	std::vector<Object> Track(std::vector<Object>& segments);
};