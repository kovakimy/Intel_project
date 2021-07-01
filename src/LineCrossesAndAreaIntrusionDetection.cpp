#include "../include/LineCrossesAndAreaIntrusionDetection.hpp"

LineCrossesAndAreaIntrusionDetection::LineCrossesAndAreaIntrusionDetection() {}

void LineCrossesAndAreaIntrusionDetection::checkLineCrosses(std::vector<BoundaryLine>& boundaryLines, 
	std::unordered_map<size_t, std::vector<cv::Point> >& activeTracks, std::vector<cv::Point> & pastSegments) {
	for (auto& trackPair : activeTracks) {
		auto objectId = trackPair.first;
		auto track = trackPair.second;
		size_t trackLen = track.size();
		if (trackLen < 2) continue;

		for (auto& bLine : boundaryLines) {
			for (size_t i = 0; i < trackLen - 1; ++i) {
				cv::Point p0 = track[i];
				cv::Point p1 = track[i + 1];

				if (segmentsIntersect(p0, p1, bLine.p0, bLine.p1) && !vectorContainsSegment(pastSegments, p0, p1)) {
					pastSegments.push_back(p0);
					pastSegments.push_back(p1);
					if (computeAngle(p0, p1, bLine.p0, bLine.p1) < 180) {
						bLine.count1 += 1;
					}
					else {
						bLine.count2 += 1;
					}
				}
			}
		}
	}
}

void LineCrossesAndAreaIntrusionDetection::checkAreaIntrusion(std::vector<Area>& areas, cv::detail::tracking::tbm::TrackedObjects& detections) {
	for (auto& area : areas) {
		area.count = 0;
		for (auto& detection : detections) {
			cv::Point p0 = (detection.rect.tl() + detection.rect.br()) / 2;
			if (cv::pointPolygonTest(area.contour, p0, false) >= 0) {
				area.count++;
			}
		}
	}
}

bool vectorContainsSegment(const std::vector<cv::Point>& segments, const cv::Point & p0, const cv::Point& p1) {
	if (segments.size() > 2) {
		for (size_t i = 1; i < segments.size() - 1; ++i) {
			if (segments[i - 1] == p0 && segments[i] == p1 || segments[i] == p0 && segments[i + 1] == p1) {
				return true;
			}
		}
	}
	return false;
}