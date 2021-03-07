#include <iostream>
#include "../include/detector.hpp"
#include "../include/ObjectTracker.hpp"
#include "../include/reid_network.hpp"
#include "../include/LineCrossingDetection.hpp"
#include "../include/AreaIntrusionDetection.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#define str "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define pd 60

void progressBar(double progress) {
    int val = (int)(progress * 100);
    int left = (int)(progress * pd);
    int rigth = pd - left;
    printf("\r%3d%% [%.*s%*s]", val, left, str, rigth, "");
    fflush(stdout);
}

cv::Mat draw_bbox(cv::Mat &frame, std::vector<DetectionObject>& objects)
{
    for (int i = 0; i < objects.size(); i++)
    {
        if (objects[i].confidence > 0.75) 
            cv::rectangle(frame, cv::Rect(cv::Point(objects[i].xmin, objects[i].ymin), cv::Point(objects[i].xmax, objects[i].ymax)), cv::Scalar(225, 0, 0));
    }
    return frame;
}

cv::Mat crop(cv::Mat& img, int xmin, int ymin, int xmax, int ymax)
{
	cv::Rect roi;
    roi.x = xmin;
    roi.y = ymax;
    roi.width = abs(xmax - xmin);
    roi.height = abs(ymin - ymax);

	cv::Mat crop = img(roi);
	return crop;
}

std::vector<Object>& turnToObject(std::vector<DetectionObject>& detections, cv::Mat& frame, ReidentificationNet& ri)
{
	std::vector<Object> Objects;//вектор объектов из части areas_and_blines
	for (auto detect : detections)
	{
		cv::Mat croped = crop(frame, detect.xmin, detect.ymin, detect.xmax, detect.ymax);
		std::vector<float> features = ri.doEverything(croped);//метод Антона Коркунова
		std::vector<cv::Point> position; //вектор координат сначала координата min, вторая max
		cv::Point tmp1(detect.xmin, detect.ymin);
		cv::Point tmp2(detect.xmax, detect.ymax);
		position.push_back(tmp1);
		position.push_back(tmp2);
		Object tmpObject(position, features, -1);
		Objects.push_back(tmpObject);

	}
	return Objects;
}


int main()
{
    InferenceEngine::Core ie;
    InferenceEngine::Core reid_ie;
    std::string FLAGS_m = "../models/pedestrian-detection-adas-0002.xml";
    std::string FLAGS_c = "../models/pedestrian-detection-adas-0002.bin";
    std::string FLAGS_v = "../media/people-detection.mp4";
	std::string FLAGS_mReidentification="../models/person-reidentification-retail-0286.xml";
	std::string FLAGS_cReidentification="../models/person-reidentification-retail-0286.bin";

    Detector detector(FLAGS_m, FLAGS_c);
    ReidentificationNet ri(FLAGS_mReidentification, FLAGS_cReidentification, ie);

    int frame_counter = 1;
    cv::Mat frame;
    cv::Mat result;
    cv::VideoCapture capture(FLAGS_v);

    double frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    double frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter out("../media/out.avi",
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height), true);

	ObjectTracker NewTracker(FLT_MAX, FLT_MAX);

	std::vector<cv::Point> contour = { cv::Point(200, 200), cv::Point(500, 180), cv::Point(600, 400), cv::Point(300, 300), cv::Point(100, 360) };
	std::vector<Area> areas = { Area(contour) };
	AreaIntrusionDetection areaDetectorAndDrawer(areas);

	std::vector<BoundaryLine> boundaryLines = { BoundaryLine(cv::Point(300, 40), cv::Point(200,400)), BoundaryLine(cv::Point(440, 40), cv::Point(700,400)) };
	LineCrossingDetection bLinesDetectorAndDrawer(boundaryLines);

    std::cout << "Progress bar..." << std::endl;
    while (frame_counter < capture.get(cv::CAP_PROP_FRAME_COUNT))
    {
        progressBar((frame_counter + 1) / capture.get(cv::CAP_PROP_FRAME_COUNT));
        capture >> frame;
        if (frame.empty())
        {
            frame_counter++;
            continue;
        }
        std::vector<DetectionObject> detections = detector.getDetections(frame);
		std::vector<Object> Objects;
		Objects = turnToObject(detections, frame, ri);
		//tracking
		std::vector<Object> NewObjects;

		NewObjects=NewTracker.Track(Objects);

		//drawing
		areaDetectorAndDrawer.checkAreaIntrusion(NewObjects);
		areaDetectorAndDrawer.drawAreas(frame);

		bLinesDetectorAndDrawer.checkLineCrosses(NewObjects);
		bLinesDetectorAndDrawer.drawBoundaryLines(frame);

        result = draw_bbox(frame, detections);
        out.write(result);
        frame_counter++;
        capture >> frame;
    }

    std::cout << std::endl;
    std::cout << "Completed";
    return 0;
}