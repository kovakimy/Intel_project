#include "../include/Drawer.hpp"

Drawer::Drawer(size_t numColors) {
	colors = generateColors(numColors);
}

void Drawer::drawBboxWithId(cv::Mat& frame, cv::detail::tracking::tbm::TrackedObjects & detections) {
	for (const auto& detection : detections)
	{
		cv::Scalar& color = colors[detection.object_id % colors.size()];
		std::string text = std::to_string(detection.object_id) +
			" conf: " + std::to_string(detection.confidence);
		cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_PLAIN, 1, color, 2);
		cv::rectangle(frame, detection.rect, color, 2);
	}
}
void Drawer::drawBoundaryLines(cv::Mat& frame, std::vector<BoundaryLine>& boundaryLines) {
	for (auto& bLine : boundaryLines) {
		line(frame, bLine.p0, bLine.p1, bLine.color, bLine.lineThinkness);
		putText(frame, std::to_string(bLine.count1), bLine.p0, cv::FONT_HERSHEY_PLAIN, bLine.textSize, bLine.textColor, bLine.lineThinkness);
		putText(frame, std::to_string(bLine.count2), bLine.p1, cv::FONT_HERSHEY_PLAIN, bLine.textSize, bLine.textColor, bLine.lineThinkness);
	}
}
void Drawer::drawAreas(cv::Mat& frame, std::vector<Area>& areas) {
	for (auto& area : areas) {
		if (area.count > 0) {
			area.color = cv::Scalar(0, 0, 255);
		}
		else {
			area.color = cv::Scalar(255, 0, 0);
		}
		polylines(frame, area.contour, true, area.color, 4);
		putText(frame, std::to_string(area.count), area.contour[0], cv::FONT_HERSHEY_PLAIN, 4, area.color, 2);
	}
}