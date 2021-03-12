#pragma once

#include "../include/Context.hpp"

class LineCrossesAndAreaIntrusionDetection {
public:
	explicit LineCrossesAndAreaIntrusionDetection();

	void checkLineCrosses(std::vector<BoundaryLine>& boundaryLines, std::vector<Object>& objects);
	void checkAreaIntrusion(std::vector<Area>& areas, std::vector<Object>& objects);
};