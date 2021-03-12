#pragma once

#include "../include/Context.hpp"

class Drawer {
public:
	explicit Drawer();

	void drawTrajectory(cv::Mat& frame, std::vector<Object>& objects);
	void drawBboxWithId(cv::Mat& frame, std::vector<Object>& objects);
	void drawBoundaryLines(cv::Mat& frame, std::vector<BoundaryLine>& boundaryLines);
	void drawAreas(cv::Mat& frame, std::vector<Area>& areas);
};