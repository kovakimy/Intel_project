#include "../include/ObjectDetector.hpp"
#include "../include/ReidNetwork.hpp"
#include "../include/LineCrossesAndAreaIntrusionDetection.hpp"
#include "../include/Drawer.hpp"
#include "../include/Kalman.hpp"
#include "../include/ReidDescriptor.hpp"


#define str "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define pd 60

void progressBar(double progress) {
	int val = (int)(progress * 100);
	int left = (int)(progress * pd);
	int rigth = pd - left;
	printf("\r%3d%% [%.*s%*s]", val, left, str, rigth, "");
	fflush(stdout);
}
std::string FLAGS_mReidentification = "C:\\Users\\AMusatov\\Intel_project\\models\\person-reidentification-retail-0288.xml";
std::string FLAGS_cReidentification = "C:\\Users\\AMusatov\\Intel_project\\models\\person-reidentification-retail-0288.bin";

InferenceEngine::Core ie;

cv::Ptr<cv::detail::tracking::tbm::ITrackerByMatching> createTrackerByMatchingWithStrongDescriptor();

cv::Ptr<cv::detail::tracking::tbm::ITrackerByMatching> createTrackerByMatchingWithStrongDescriptor() {
	cv::detail::tracking::tbm::TrackerParams params;

	cv::Ptr<cv::detail::tracking::tbm::ITrackerByMatching> tracker = createTrackerByMatching(params);

	std::shared_ptr<cv::detail::tracking::tbm::IImageDescriptor> descriptor_strong =
		std::make_shared<ReidDescriptor>(FLAGS_mReidentification, FLAGS_cReidentification, ie);
	std::shared_ptr<cv::detail::tracking::tbm::IDescriptorDistance> distance_strong =
		std::make_shared<cv::detail::tracking::tbm::CosDistance>(descriptor_strong->size());
	std::shared_ptr<cv::detail::tracking::tbm::IImageDescriptor> descriptor_fast =
		std::make_shared<cv::detail::tracking::tbm::ResizedImageDescriptor>(
			cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
	std::shared_ptr<cv::detail::tracking::tbm::IDescriptorDistance> distance_fast =
		std::make_shared<cv::detail::tracking::tbm::MatchTemplateDistance>();

	tracker->setDescriptorFast(descriptor_fast);
	tracker->setDistanceFast(distance_fast);

	tracker->setDescriptorStrong(descriptor_strong);
	tracker->setDistanceStrong(distance_strong);

	return tracker;
}

void setUpAreasAndBoundaryLines(cv::Mat& frame, size_t countAreas, size_t countLines, std::vector<Area>& areas, std::vector<BoundaryLine>& boundaryLines) {
	Drawer drawer;
	std::string areasWindowName = "Setting up areas",
		linesWindowName = "Setting up boundary lines";

	for (size_t i = 0; i < countAreas; ++i) {
		std::cout << "Select multiple points for the area " << i << " and press any key."<< std::endl;
		Parameters params;
		params.mode = 0; // setting areas
		params.frame = frame;
		params.windowName = areasWindowName;
		cv::namedWindow(areasWindowName, 1);
		cv::setMouseCallback(areasWindowName, callback, static_cast<void*>(&params));
		cv::imshow(areasWindowName, frame);
		cv::waitKey(0);
		areas.push_back(Area(params.areaContour));
		drawer.drawAreas(frame, areas);
	}
	cv::destroyWindow("Setting up areas");

	for (size_t i = 0; i < countLines; ++i) {
		std::cout << "Select 2 points for the boundary line " << i << " and press any key." << std::endl;
		Parameters params;
		params.mode = 1; // setting lines
		params.frame = frame;
		params.windowName = linesWindowName;
		cv::namedWindow(linesWindowName, 1);
		cv::setMouseCallback(linesWindowName, callback, static_cast<void*>(&params));
		cv::imshow(linesWindowName, frame);
		cv::waitKey(0);
		boundaryLines.push_back(BoundaryLine(params.linePoints[0], params.linePoints[1]));
		drawer.drawBoundaryLines(frame, boundaryLines);
	}
	std::cout << "Press any key to start a demo." << std::endl;
	cv::imshow(linesWindowName, frame);
	cv::waitKey(0);
	cv::destroyWindow("Setting up boundary lines");
}

int main() {
	std::string FLAGS_m = "C:\\Users\\AMusatov\\Intel_project\\models\\person-detection-0202.xml";
	std::string FLAGS_c = "C:\\Users\\AMusatov\\Intel_project\\models\\person-detection-0202.bin";
	std::string FLAGS_v = "C:\\Users\\AMusatov\\Intel_project\\media\\people-detection.mp4";

	Detector detector(FLAGS_m, FLAGS_c, ie);

	int frame_counter = 1;
	float R = 1e-4;
	cv::Mat frame;
	cv::Mat result;
	cv::VideoCapture capture(FLAGS_v);

	double frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	double frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter out("C:\\Users\\AMusatov\\Intel_project\\media\\out_detect11.avi",
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height), true);
	
	cv::Ptr<cv::detail::tracking::tbm::ITrackerByMatching> tracker = createTrackerByMatchingWithStrongDescriptor();

	int frame_step = 1;
	int64 time_total = 0;

	LineCrossesAndAreaIntrusionDetection solver = LineCrossesAndAreaIntrusionDetection();
	Drawer drawer;
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
	cv::detail::tracking::tbm::TrackedObjects detections;
	std::unordered_map<size_t, std::vector<cv::Point> > activeTracks;
	std::vector<cv::Point> pastSegments;

	while (frame_counter < capture.get(cv::CAP_PROP_FRAME_COUNT))
	{
		progressBar(2*(frame_counter + 1) / capture.get(cv::CAP_PROP_FRAME_COUNT));
		capture >> frame;
		if (frame.empty())
		{
			frame_counter++;
			continue;
		}
		frame_counter++;
		
		if (frame_counter % frame_step == 0)
		{
			int64 frame_time = cv::getTickCount();

			detections = detector.getDetections(frame, frame_counter);
			// timestamp in milliseconds
			uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / 10 * frame_counter);
			tracker->process(frame, detections, cur_timestamp);

			frame_time = cv::getTickCount() - frame_time;
			time_total += frame_time;

			solver.checkAreaIntrusion(areas, detections);
			for (const auto& detection : detections) {
				cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 2);
			}
		}

		detections = tracker->trackedDetections();
		activeTracks = tracker->getActiveTracks();
		
		if (frame_counter % frame_step > 0) solver.checkAreaIntrusion(areas, detections);
		solver.checkLineCrosses(boundaryLines, activeTracks, pastSegments);
		
		drawer.drawBboxAndTrajectory(frame, detections, activeTracks);
		drawer.drawBoundaryLines(frame, boundaryLines);
		drawer.drawAreas(frame, areas);

		out.write(frame);
	}

	double s = frame_counter / (time_total / cv::getTickFrequency());
	std::cout << std::endl << s << std::endl;
	std::cout << "Completed";
	return 0;
}