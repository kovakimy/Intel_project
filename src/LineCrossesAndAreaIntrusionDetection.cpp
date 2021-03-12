#include "../include/LineCrossesAndAreaIntrusionDetection.hpp"

LineCrossesAndAreaIntrusionDetection::LineCrossesAndAreaIntrusionDetection() {}

void LineCrossesAndAreaIntrusionDetection::checkLineCrosses(std::vector<BoundaryLine>& boundaryLines, std::vector<Object>& objects) {
	for (auto& object : objects) {
		size_t trajLen = object.trajectory.size();
		if (trajLen < 2) continue;

		cv::Point p0_traj = object.trajectory[trajLen - 1];
		cv::Point p1_traj = object.trajectory[trajLen - 2];

		for (auto& bLine : boundaryLines) {
			if (segmentsIntersect(p0_traj, p1_traj, bLine.p0, bLine.p1))
				bLine.count1 += 1;
			else
				bLine.count2 += 1;
		}
	}
}

void LineCrossesAndAreaIntrusionDetection::checkAreaIntrusion(std::vector<Area>& areas, std::vector<Object>& objects) {
	for (auto& area : areas) {
		area.count = 0;
		for (auto& obj : objects) {
			cv::Point p0 = (obj.pos[0] + obj.pos[1]) / 2;
			if (cv::pointPolygonTest(area.contour, p0, false) >= 0) {
				area.count++;
			}
		}
	}
}