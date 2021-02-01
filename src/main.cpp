#include <iostream>
#include "../include/detector.hpp"
#include <opencv2/highgui.hpp>

cv::Mat draw_bbox(cv::Mat frame, std::vector<DetectionObject>& objects) {
	for (int i = 0; i < objects.size(); i++) {
		if (objects[i].confidence > 0.5) 
			cv::rectangle(frame, cv::Rect(cv::Point(objects[i].xmin, objects[i].ymin), cv::Point(objects[i].xmax, objects[i].ymax)), cv::Scalar(225, 0, 0));
	}
	return frame;
}

int main()
{
	InferenceEngine::Core ie;
	std::string FLAGS_m = "C://Users//kovakimy//models//pedestrian-detection-adas-0002.xml";
	std::string configPath = "C://Users//kovakimy//models//pedestrian-detection-adas-0002.bin";
	std::string videoPath = "C://Users//kovakimy//models//people-detection.h264";

	Detector detector(FLAGS_m, configPath); 

	cv::Mat frame;
	cv::VideoCapture capture(videoPath);

	capture >> frame;

	while (capture.get(cv::CAP_PROP_POS_FRAMES) != capture.get(cv::CAP_PROP_POS_FRAMES)) {
		detector.getDetections(frame);
		capture >> frame;
	}
	
	return 0;
}