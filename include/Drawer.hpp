#pragma once

#include "../include/Context.hpp"

class Drawer {
public:
	explicit Drawer(size_t numColors = 300);
	std::vector<cv::Scalar> colors;

	void drawBboxWithId(cv::Mat& frame, cv::detail::tracking::tbm::TrackedObjects& detections);
	void drawBoundaryLines(cv::Mat& frame, std::vector<BoundaryLine>& boundaryLines);
	void drawAreas(cv::Mat& frame, std::vector<Area>& areas);
};