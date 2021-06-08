#include "../include/ObjectDetector.hpp"
#include "../include/ObjectTracker.hpp"
#include "../include/ReidNetwork.hpp"
#include "../include/LineCrossesAndAreaIntrusionDetection.hpp"
#include "../include/Drawer.hpp"
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

void setUpAreasAndBoundaryLines(cv::Mat& frame, size_t countAreas, size_t countLines, std::vector<Area>& areas, std::vector<BoundaryLine>& boundaryLines) {
	Drawer drawer;

	for (size_t i = 0; i < countAreas; ++i) {
		std::cout << "Select multiple points for the area " << i << " and press any key."<< std::endl;
		Parameters params;
		params.mode = 0; // => setting areas
		cv::namedWindow("Setting up areas", 1);
		cv::setMouseCallback("Setting up areas", callback, static_cast<void*>(&params));
		cv::imshow("Setting up areas", frame);
		cv::waitKey(0);
		cv::destroyWindow("Setting up areas");
		areas.push_back(Area(params.areaContour));
		drawer.drawAreas(frame, areas);
	}

	for (size_t i = 0; i < countLines; ++i) {
		std::cout << "Select 2 points for the boundary line " << i << " and press any key." << std::endl;
		Parameters params;
		params.mode = 1; // => setting lines
		cv::namedWindow("Setting up boundary lines", 1);
		cv::setMouseCallback("Setting up boundary lines", callback, static_cast<void*>(&params));
		cv::imshow("Setting up boundary lines", frame);
		cv::waitKey(0);
		cv::destroyWindow("Setting up boundary lines");
		boundaryLines.push_back(BoundaryLine(params.linePoints[0], params.linePoints[1]));
		drawer.drawBoundaryLines(frame, boundaryLines);
	}
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
	

	int parameters_is_set_up = false;
	int frame_counter = 1;
	cv::Mat frame;
	cv::Mat result;
	cv::VideoCapture capture(FLAGS_v);

	double frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	double frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter out("../media/out11.avi",
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height), true);

	//ObjectTracker NewTracker(FLT_MAX, FLT_MAX);
	ObjectTracker NewTracker(0.42, 0.42);  // (0.6, 0.25) (0.8, 0.8)

	LineCrossesAndAreaIntrusionDetection solver = LineCrossesAndAreaIntrusionDetection();
	Drawer drawer = Drawer();

	std::vector<Area> areas;
	std::vector<BoundaryLine> boundaryLines;
	size_t countAreas, countLines;
	std::cout << "Input a number of areas: ";
	std::cin >> countAreas;
	std::cout << "Input a number of lines: ";
	std::cin >> countLines;

	capture >> frame;
	if (!frame.empty()) {
		setUpAreasAndBoundaryLines(frame, countAreas, countLines, areas, boundaryLines);
		frame_counter++;
	}

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
		std::vector<Object> objects;

		objects = turnToObject(detections, frame, ri);
		//tracking

		objects = NewTracker.Track(objects);

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