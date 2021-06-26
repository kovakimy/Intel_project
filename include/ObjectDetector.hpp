#include <string>
#include <ie_core.hpp>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/tracking/tracking_by_matching.hpp>

struct DetectionObject
{
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(int class_id, double score, double x0, double y0, double x1, double y1)
    {
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
private:
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork cnnNetwork;
    std::string modelPath, configPath;
    InferenceEngine::ICNNNetwork::InputShapes inputShapes;
    std::string inName, outName;
    int out_blob_h, out_blob_w;
    InferenceEngine::InferRequest inferRequest;
public:
    Detector(const std::string &, const std::string &, const InferenceEngine::Core&);
    void createRequest(const cv::Mat&);
    cv::detail::tracking::tbm::TrackedObjects getDetections(const cv::Mat &, int);
};