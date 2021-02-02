#include <iostream>
#include "../include/detector.hpp"
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

int main()
{
    InferenceEngine::Core ie;
    std::string FLAGS_m = "C://Users//kovakimy//models//pedestrian-detection-adas-0002.xml";
    std::string FLAGS_c = "C://Users//kovakimy//models//pedestrian-detection-adas-0002.bin";
    std::string FLAGS_v = "C://Users//kovakimy//models//people-detection.mp4";

    Detector detector(FLAGS_m, FLAGS_c);

    int frame_counter = 1;
    cv::Mat frame;
    cv::Mat result;
    cv::VideoCapture capture(FLAGS_v);

    double frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    double frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter out("C:/Users/kovakimy/OneDrive - Intel Corporation/Pictures/intel_project/demos/build/out.avi",
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height), true);

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
        result = draw_bbox(frame, detections);
        out.write(result);
        frame_counter++;
        capture >> frame;
    }
    std::cout << std::endl;
    std::cout << "Completed";
    return 0;
}