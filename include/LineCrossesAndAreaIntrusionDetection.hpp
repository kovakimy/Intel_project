#pragma once

#include "../include/Context.hpp"

class LineCrossesAndAreaIntrusionDetection {
public:
	explicit LineCrossesAndAreaIntrusionDetection();

	void checkLineCrosses(std::vector<BoundaryLine>& boundaryLines, std::unordered_map<size_t, std::vector<cv::Point> >& activeTracks, std::vector<cv::Point>& pastSegments);
	void checkAreaIntrusion(std::vector<Area>& areas, cv::detail::tracking::tbm::TrackedObjects& detections);
};


bool vectorContainsSegment(const std::vector<cv::Point>& track, const cv::Point& p0, const cv::Point& p1);