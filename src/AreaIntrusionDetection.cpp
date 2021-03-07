#include "../include/AreaIntrusionDetection.hpp"

AreaIntrusionDetection::AreaIntrusionDetection(std::vector<Area>& _areas) {
	areas = _areas;
};

void AreaIntrusionDetection::checkAreaIntrusion(std::vector<Object>& objs) {
	for (auto& area : areas) {
		area.count = 0;
		for (auto& obj : objs) {
			cv::Point p0 = (obj.pos[0] + obj.pos[1]) / 2;
			if (cv::pointPolygonTest(area.contour, p0, false) >= 0) {
				area.count++;
			}
		}
	}
}

void AreaIntrusionDetection::drawAreas(const cv::Mat& img) {
	for (auto& area : areas) {
		if (area.count > 0) {
			area.color = cv::Scalar(0, 0, 255);
		}
		else {
			area.color = cv::Scalar(255, 0, 0);
		}
		polylines(img, area.contour, true, area.color, 4);
		putText(img, std::to_string(area.count), area.contour[0], cv::FONT_HERSHEY_PLAIN, 4, area.color, 2);
	}
}
