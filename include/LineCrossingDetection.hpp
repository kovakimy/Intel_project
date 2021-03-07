#pragma once

#include "Context.hpp"


class LineCrossingDetection {
private:
	std::vector<BoundaryLine> boundaryLines;
public:
	explicit LineCrossingDetection(std::vector<BoundaryLine>& boundaryLines);
	void checkLineCrosses(std::vector<Object>& objects);
	void drawBoundaryLines(cv::Mat& image) const;
};