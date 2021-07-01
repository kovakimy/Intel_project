#pragma once

#include "../include/Context.hpp"

class LineCrossesAndAreaIntrusionDetection {
public:
	explicit LineCrossesAndAreaIntrusionDetection();

	void checkLineCrosses(std::vector<BoundaryLine>& boundaryLines, std::unordered_map<size_t, std::vector<cv::Point> >& activeTracks, std::vector<size_t>& oldIds);
	void checkAreaIntrusion(std::vector<Area>& areas, cv::detail::tracking::tbm::TrackedObjects& detections);
};