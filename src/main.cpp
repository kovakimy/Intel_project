#include "LineCrossingDetection.hpp"
#include "AreaIntrusionDetection.hpp"
#include <iostream>

int main() {
	Point p1(300, 40);
	Point p2(20, 400);
	Point p3(440, 40);
	Point p4(600, 400);

	std::vector<Point> contur = {Point(200, 200), Point(500, 180), Point(600, 400), Point(300, 300), Point(100, 360) };
	std::vector<Area> areas = { Area(contur) };
	AreaIntrusionDetection areaDetAndDraw(areas);

	std::vector<BoundaryLine> boundaryLines = { BoundaryLine(p1,p2), BoundaryLine(p3,p4) };
	LineCrossingDetection bLinesDetAndDraw(boundaryLines);

	std::cout << segmentsIntersect(p1, p2, p3, p4) << std::endl;
	std::cout << computeAngle(p1, p2, p3, p4);

	Mat img(500, 700, CV_8UC3, Scalar(255, 255, 255));
	bLinesDetAndDraw.drawBoundaryLines(img);
	areaDetAndDraw.drawAreas(img);

	imshow("Img", img);

	waitKey(0);
}