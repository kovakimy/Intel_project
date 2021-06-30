#include "../include/Context.hpp"

using namespace cv;

Object::Object(std::vector<Point>& pos, std::vector<float>& feature, int id)
	: pos(pos), feature(feature), id(id), time(std::chrono::steady_clock::now()) 
{ 
	for (std::size_t i = 0; i < 4; ++i)
	{
		P.push_back(std::vector<float>(4, 0));
		x.push_back(std::vector<float>(1, 0));
	}
	P[0][0] = 1000;
	P[1][1] = 1000;
	P[2][2] = 1000;
	P[3][3] = 1000;
};

Area::Area(std::vector<Point>& contour)
	: contour(contour), count(0) {};

BoundaryLine::BoundaryLine(const Point& p0, const Point& p1) {
	this->p0 = p0;
	this->p1 = p1;
	color = Scalar(0, 255, 255);
	textColor = Scalar(0, 255, 255);
	textSize = 4;
	lineThinkness = 4;
	textThinkness = 2;
	count1 = 0;
	count2 = 0;
};

inline Point vectorize(const Point& p1, const Point& p2) {
	return p2 - p1;
}

inline int direction(const Point& p1, const Point& p2, const Point& p3) {
	Point p1p3 = vectorize(p1, p3);
	Point p1p2 = vectorize(p1, p2);

	return trunc(p1p3.cross(p1p2));
};

inline bool onSegment(const Point& p1, const Point& p2, const Point& p3) {
	return std::min(p1.x, p2.x) <= p3.x && p3.x <= std::max(p1.x, p2.x) &&
		std::min(p1.y, p2.y) <= p3.y && p3.y <= std::max(p1.y, p2.y);
};

bool segmentsIntersect(const Point& p1, const Point& p2, const Point& p3, const Point& p4) {
	int d1 = direction(p3, p4, p1);
	int d2 = direction(p3, p4, p2);
	int d3 = direction(p1, p2, p3);
	int d4 = direction(p1, p2, p4);

	bool intersect = ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
		((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));

	bool pointOnSegment = (d1 == 0 && onSegment(p3, p4, p1)) || (d2 == 0 && onSegment(p3, p4, p2)) ||
		(d3 == 0 && onSegment(p1, p2, p3)) || (d4 == 0 && onSegment(p1, p2, p4));

	return intersect || pointOnSegment;
};

double computeAngle(const Point& p1, const Point& p2, const Point& p3, const Point& p4) {
	Point vec1 = vectorize(p1, p2);
	Point vec2 = vectorize(p3, p4);
	
	double prod = static_cast<double>(vec1.dot(vec2));
	double normProd = norm(vec1) * norm(vec2);
	double radianAngle = acos(prod / normProd);
	double degreeAngle = 180 * radianAngle / M_PI;
	degreeAngle = trunc(degreeAngle * PRECISION) / PRECISION;
	
	return vec1.cross(vec2) < 0 ? degreeAngle : 360 - degreeAngle;
}


void callback(int event, int x, int y, int flag, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		Parameters* params = static_cast<Parameters*>(userdata);
		if (params->mode == 0) {
			cv::Point p(x, y);
			params->areaContour.push_back(p);
			size_t countourSize = params->areaContour.size();
			if (countourSize > 1) {
				cv::line(params->frame, params->areaContour[countourSize - 1], params->areaContour[countourSize - 2], cv::Scalar(0, 0, 255), 2);
			}
			else {
				cv::circle(params->frame, p, 2, cv::Scalar(0, 0, 255), cv::FILLED);
			}
			cv::imshow(params->windowName, params->frame);
			std::cout << "A point chosen for the area: (" << x << ", " << y << ")" << std::endl;
		}
		else if (params->mode == 1) {
			cv::Point p(x, y);
			params->linePoints.push_back(p);
			size_t numPoints = params->linePoints.size();
			if (numPoints > 1) {
				cv::line(params->frame, params->linePoints[numPoints - 1], params->linePoints[numPoints - 2], cv::Scalar(0, 0, 255), 2);
			}
			else {
				cv::circle(params->frame, p, 2, cv::Scalar(0, 0, 255), cv::FILLED);
			}
			cv::circle(params->frame, p, 1, cv::Scalar(0, 0, 255), cv::FILLED);
			cv::imshow(params->windowName, params->frame);
			std::cout << "A point chosen for the boundary line: (" << x << ", " << y << ")" << std::endl;
		}	
	}
}

std::vector<cv::Scalar> generateColors(size_t size) {
	std::vector<cv::Scalar> colors;
	std::random_device dev;
	std::mt19937 range(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);

	for (size_t i = 0; i < size; ++i) {
		colors.push_back(cv::Scalar(dist(range), dist(range), dist(range)));
	}

	return colors;
}