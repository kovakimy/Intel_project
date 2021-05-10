#include <iostream>
#include "../include/ObjectDetector.hpp"
#include "../include/ObjectTracker.hpp"
#include "../include/ReidNetwork.hpp"
#include "../include/LineCrossesAndAreaIntrusionDetection.hpp"
#include "../include/Drawer.hpp"
#include "../include/Kalman.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

#define str "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define pd 60

void progressBar(double progress) {
	int val = (int)(progress * 100);
	int left = (int)(progress * pd);
	int rigth = pd - left;
	printf("\r%3d%% [%.*s%*s]", val, left, str, rigth, "");
	fflush(stdout);
}

//for obj in objects :
//if len(obj.trajectory) > 1:
//cv2.polylines(img, np.array([obj.trajectory], np.int32), False, (0, 0, 0), 4)

cv::Mat crop(cv::Mat& img, int xmin, int ymin, int xmax, int ymax) {
	//	cv::Rect roi;
	//	roi.x = xmin;
	//	roi.y = ymin;//img.size().height - ymax;
		//roi.width = abs(xmax - xmin) - 2;
		//roi.height = ymax - ymin;

		//cv::Mat crop = img(roi);

	int width = xmax - xmin, height = ymax - ymin;

	cv::Mat ROI(img, cv::Rect(xmin, ymin, width, height));

	cv::Mat crop;

	// Copy the data into new matrix
	ROI.copyTo(crop);

	//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
	//imshow("Display window", crop);
 //   cv::waitKey(0);
	return crop;
}

std::vector<Object> turnToObject(std::vector<DetectionObject>& detections, cv::Mat& frame, ReidentificationNet& ri) {
	std::vector<Object> Objects;//vector of objects from areas_and_blines
	for (auto detect : detections)
	{
		cv::Mat croped = crop(frame, detect.xmin, detect.ymin, detect.xmax, detect.ymax);
		std::vector<float> features = ri.doEverything(croped);//common method for the work process of the reid network
		std::vector<cv::Point> position; //coordinates vector, at first min and after max coordinate
		cv::Point tmp1(detect.xmin, detect.ymin);
		cv::Point tmp2(detect.xmax, detect.ymax);
		position.push_back(tmp1);
		position.push_back(tmp2);
		Object tmpObject(position, features, -1);
		Objects.push_back(tmpObject);
	}
	return Objects;
}


int main() {
	InferenceEngine::Core ie;
	InferenceEngine::Core reid_ie;
	std::string FLAGS_m = "../models/person-detection-0202.xml";
	std::string FLAGS_c = "../models/person-detection-0202.bin";
	std::string FLAGS_v = "../media/people-detection.mp4";
	std::string FLAGS_mReidentification = "../models/person-reidentification-retail-0286.xml";
	std::string FLAGS_cReidentification = "../models/person-reidentification-retail-0286.bin";

	Detector detector(FLAGS_m, FLAGS_c, ie);
	ReidentificationNet ri(FLAGS_mReidentification, FLAGS_cReidentification, ie);

	int frame_counter = 1;
	float R = 1e-4;
	cv::Mat frame;
	cv::Mat result;
	cv::VideoCapture capture(FLAGS_v);

	double frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	double frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter out("../media/out_detect.avi",
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height), true);

	ObjectTracker NewTracker(0.42, 0.42);  // (0.6, 0.25) (0.8, 0.8)
	std::string trackingAlg = "KCF";
	MultiTracker trackers;
	std::vector<Ptr<Tracker> > algorithms;

	auto solver = LineCrossesAndAreaIntrusionDetection();
	auto drawer = Drawer();

	std::vector<cv::Point> contour = { cv::Point(200, 200), cv::Point(500, 180), cv::Point(600, 400), cv::Point(300, 300), cv::Point(100, 360) };
	std::vector<Area> areas = { Area(contour) };

	std::vector<BoundaryLine> boundaryLines = { BoundaryLine(cv::Point(217, 40), cv::Point(50,400)), BoundaryLine(cv::Point(440, 40), cv::Point(700,400)) };

	std::cout << "Progress bar..." << std::endl;
	std::vector<Object> objects;
	while (frame_counter < capture.get(cv::CAP_PROP_FRAME_COUNT))
	{
		progressBar((frame_counter + 1) / capture.get(cv::CAP_PROP_FRAME_COUNT));
		capture >> frame;
		if (frame.empty())
		{
			frame_counter++;
			continue;
		}
		
		if (frame_counter % 4 == 0)
		{
			std::vector<Object> tmpObjects;
			std::vector<DetectionObject> detections = detector.getDetections(frame);
			tmpObjects = turnToObject(detections, frame, ri);
			objects = NewTracker.Track(tmpObjects, algorithms);

			trackers.add(algorithms, frame, objects); // NOT OBJECTS, BUT RECTANGLES (vector<Rect2D>);
		}
		trackers.update(frame);
		
		/*std::vector<DetectionObject> detections = detector.getDetections(frame);

		std::vector<Object> objects;

		objects = turnToObject(detections, frame, ri);
		////tracking

		objects = NewTracker.Track(objects);
		
		if ((frame_counter % 3) == 0 || objects.size() == 0)
		{
			//std::vector<Object> objects;
			std::vector<DetectionObject> detections = detector.getDetections(frame);
			objects = turnToObject(detections, frame, ri);
			////tracking

			objects = NewTracker.Track(objects);

			for (auto& obj : objects)
			{
				kalman(obj.x, obj.P, obj.trajectory.back().x, obj.trajectory.back().y, R);
			}
		}
		else
		{
			for (auto& obj : objects)
			{
				obj.trajectory.push_back(kalman(obj.x, obj.P, obj.trajectory.back().x, obj.trajectory.back().y, R));
			}
		}*/
		////check
		solver.checkAreaIntrusion(areas, objects);
		solver.checkLineCrosses(boundaryLines, objects);

		////drawing
		drawer.drawBboxWithId(frame, objects);
		drawer.drawTrajectory(frame, objects);
		drawer.drawBoundaryLines(frame, boundaryLines);
		drawer.drawAreas(frame, areas);

		out.write(frame);
		//	if (detections.size() > 2) {
		//	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
		 //   imshow("Display window", frame);
		//    cv::waitKey(0);
		//	}
		frame_counter++;
		capture >> frame;
	}

	std::cout << std::endl;
	std::cout << "Completed";
	return 0;
}