#include "LineCrossesAndAreaIntrusionDetection.hpp"

LineCrossesAndAreaIntrusionDetection::LineCrossesAndAreaIntrusionDetection() {}

void LineCrossesAndAreaIntrusionDetection::checkLineCrosses(std::vector<BoundaryLine>& boundaryLines, std::unordered_map<size_t, std::vector<cv::Point> >& activeTracks, std::vector<size_t>& oldIds) {
	for (auto& trackPair : activeTracks) {
		auto objectId = trackPair.first;
		auto track = trackPair.second;
		size_t trackLen = track.size();
		if (trackLen < 2 || std::find(oldIds.begin(), oldIds.end(), objectId) != oldIds.end()) continue;
		//std::cout << trackPair.first << std::endl;
		

		for (auto& bLine : boundaryLines) {
			for (size_t i = 0; i < trackLen - 1; ++i) {
				cv::Point p0 = track[i];
				cv::Point p1 = track[i + 1];

				if (segmentsIntersect(p0, p1, bLine.p0, bLine.p1)) {
					oldIds.push_back(objectId);
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