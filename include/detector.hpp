#include <string>
#include <ie_core.hpp>
#include <vector>
#include <opencv2/core.hpp>


struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(int class_id, double score, double x0, double y0, double x1, double y1) {
        this->xmin = static_cast<int>(x0);
        this->ymin = static_cast<int>(y0);
        this->xmax = static_cast<int>(x1);
        this->ymax = static_cast<int>(y1);
        this->class_id = class_id;
        this->confidence = score;
    }
};

class Detector
{
public:
	InferenceEngine::Core ie;
	InferenceEngine::CNNNetwork cnnNetwork;
	std::string modelPath, configPath;
	Detector(std::string &modelPath, std::string &configPath);
    std::vector<DetectionObject> getDetections(cv::Mat &image);
};