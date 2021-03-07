#include "../include/LineCrossingDetection.hpp"

LineCrossingDetection::LineCrossingDetection(std::vector<BoundaryLine>& boundaryLines)
	: boundaryLines(boundaryLines) { };

void LineCrossingDetection::checkLineCrosses(std::vector<Object>& objects) {
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

void LineCrossingDetection::drawBoundaryLines(cv::Mat& img) const {
	for (auto& bLine : boundaryLines) {
		line(img, bLine.p0, bLine.p1, bLine.color, bLine.lineThinkness);
		putText(img, std::to_string(bLine.count1), bLine.p0, cv::FONT_HERSHEY_PLAIN, bLine.textSize, bLine.textColor, bLine.lineThinkness);
		putText(img, std::to_string(bLine.count2), bLine.p1, cv::FONT_HERSHEY_PLAIN, bLine.textSize, bLine.textColor, bLine.lineThinkness);
	}
}