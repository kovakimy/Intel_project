#include <iostream>
#include "../include/ObjectDetector.hpp"
#include "../include/ObjectTracker.hpp"
#include "../include/ReidNetwork.hpp"
#include "../include/LineCrossesAndAreaIntrusionDetection.hpp"
#include "../include/Drawer.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define str "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define pd 60

static const char* keys =
"{ m  models_path          | <none>	| path to models                                                                }"
"{ v  video_path           | <none>	| path to models                                                                }"
"{ o  outpath              | <none>     | path to save clips                                                            }"
"{ mode                    | <none>     | mode to show result   mode=1 save result in out.avi    mode=2 show on screen  }"
"{ help h usage ?          |            | print help message                                                            }";

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
  //  cv::waitKey(0);
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


int main(int argc, char** argv) {

	cv::CommandLineParser parser(argc, argv, keys);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	cv::String FLAGS_m                 = parser.get<cv::String>("m") + "/pedestrian-detection-adas-0002.xml";
	cv::String FLAGS_c                 = parser.get<cv::String>("m") + "/pedestrian-detection-adas-0002.bin";
	cv::String FLAGS_mReidentification = parser.get<cv::String>("m") + "/person-reidentification-retail-0286.xml";
	cv::String FLAGS_cReidentification = parser.get<cv::String>("m") + "/person-reidentification-retail-0286.bin";
	cv::String FLAGS_v                 = parser.get<cv::String>("v");
	cv::String out_path                = parser.get<cv::String>("o") + "/out1.avi";
	cv::String mode                    = parser.get<cv::String>("mode");

	if (!parser.check())
	{
		parser.printErrors();
		throw "Parse error";
		return 0;
	}

	InferenceEngine::Core ie;
	InferenceEngine::Core reid_ie;

	Detector detector(FLAGS_m, FLAGS_c, ie);
	ReidentificationNet ri(FLAGS_mReidentification, FLAGS_cReidentification, ie);

	int frame_counter = 1;
	cv::Mat frame;
	cv::Mat result;
	cv::VideoCapture capture(FLAGS_v);

	double frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	double frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter out(out_path,
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height), true);

	//ObjectTracker NewTracker(FLT_MAX, FLT_MAX);
	ObjectTracker NewTracker(0.78, 0.78);
	auto solver = LineCrossesAndAreaIntrusionDetection();
	auto drawer = Drawer();

	std::vector<cv::Point> contour = { cv::Point(200, 200), cv::Point(500, 180), cv::Point(600, 400), cv::Point(300, 300), cv::Point(100, 360) };
	std::vector<Area> areas = { Area(contour) };

	std::vector<BoundaryLine> boundaryLines = { BoundaryLine(cv::Point(217, 40), cv::Point(50,400)), BoundaryLine(cv::Point(440, 40), cv::Point(700,400)) };

	std::cout << "Progress bar..." << std::endl;

        double fps = capture.get(cv::CAP_PROP_FPS);

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
		std::vector<Object> objects;

		objects = turnToObject(detections, frame, ri);
		////tracking

		objects = NewTracker.Track(objects);

		////check
		solver.checkAreaIntrusion(areas, objects);
		solver.checkLineCrosses(boundaryLines, objects);

		////drawing
		drawer.drawBboxWithId(frame, objects);
		drawer.drawTrajectory(frame, objects);
		drawer.drawBoundaryLines(frame, boundaryLines);
		drawer.drawAreas(frame, areas);
		
		if (mode == "1")
		{
			out.write(frame);
		}
		if (mode == "2") {
			cv::imshow("Display window", frame);
			if (cv::waitKey(5) >= 0)
				break;
		}
		frame_counter++;
		capture >> frame;
	}
	std::cout << std::endl;
	std::cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << std::endl;
	std::cout << std::endl;
	std::cout << "Completed";
	return 0;
}