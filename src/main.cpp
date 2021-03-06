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

cv::Mat& crop(cv::Mat& img, int xmin, int ymin, int xmax, int ymax)
{

	cv::Rect roi;
	roi.x = (xmin + xmax)/2;
	roi.y = (ymin + ymax) / 2;
	roi.width = img.size().width - (xmin + xmax);
	roi.height = img.size().height - (ymin + ymax);

	cv::Mat crop = img(roi);
	return crop;


}

std::vector<Object>& turnToObject(std::vector<DetectionObject>& detections, cv::Mat& frame, std::string FLAGS_mReidentification, std::string FLAGS_cReidentification, InferenceEngine::Core ie)
{
	ReidentificationNet ri(FLAGS_mReidentification, FLAGS_cReidentification, ie);
	std::vector<Object> Objects;//вектор объектов из части areas_and_blines
	for (auto detect : detections)
	{
		cv::Mat croped = crop(frame, detect.xmin, detect.ymin, detect.xmax, detect.ymax);
		std::vector< float> features = ri.doEverything(croped);//метод Антона Коркунова
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
    std::string FLAGS_m = "C://object-tracking-line-crossing-area-intrusion//intel//pedestrian-detection-adas-0002//FP16//pedestrian-detection-adas-0002.xml";
    std::string FLAGS_c = "C://object-tracking-line-crossing-area-intrusion//intel//pedestrian-detection-adas-0002//FP16//pedestrian-detection-adas-0002.bin";
    std::string FLAGS_v = "C://project//build//intel64//Release//people-detection.mp4";
	std::string FLAGS_mReidentification="C://object-tracking-line-crossing-area-intrusion//intel//person-reidentification-retail-0286//FP16//person-reidentification-retail-0286.xml";
	std::string FLAGS_cReidentification="C://object-tracking-line-crossing-area-intrusion//intel//person-reidentification-retail-0286//FP16//person-reidentification-retail-0286.bin";

    Detector detector(FLAGS_m, FLAGS_c);

    int frame_counter = 1;
    cv::Mat frame;
    cv::Mat result;
    cv::VideoCapture capture(FLAGS_v);

    double frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    double frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter out("C:/project/build/intel64/Release/out.avi",
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height), true);

	ObjectTracker  NewTracker(FLT_MAX, FLT_MAX);


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
		Objects = turnToObject(detections, frame, FLAGS_mReidentification, FLAGS_cReidentification, ie);
		//tracking
		std::vector<Object> NewObjects;
		
		NewObjects=NewTracker.Track(Objects);


        result = draw_bbox(frame, detections);
        out.write(result);
        frame_counter++;
        capture >> frame;
    }



    std::cout << std::endl;
    std::cout << "Completed";
    return 0;
}