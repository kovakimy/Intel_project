#pragma once

#include "Context.hpp"

class AreaIntrusionDetection {
private:
	std::vector<Area> areas;
public:
	AreaIntrusionDetection(std::vector<Area> areas);
	void checkAreaIntrusion(std::vector<Object>);
	void drawAreas(const cv::Mat& img);
};