#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <ie_core.hpp>

//model_reid = 'person-reidentification-retail-0286'

class Reidentification_results{};


class ReidentificationNet {
private:

    InferenceEngine::InferRequest request;
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork bin_network;
    InferenceEngine::ExecutableNetwork executable_network;

    std::string modelXml;
    std::string modelBin;

    InferenceEngine::InputInfo::Ptr inputInfoReid;
    InferenceEngine::DataPtr  outInfoReid;
    std::string inputName;
    std::string outputName;
    InferenceEngine::ICNNNetwork::InputShapes inputShapes;

    float width_ = 0;
    float height_ = 0;

public:
    ReidentificationNet(const std::string&, const std::string&,
        const InferenceEngine::Core&);

    InferenceEngine::InputInfo::Ptr getInput();

    InferenceEngine::DataPtr getOutput();

    InferenceEngine::ICNNNetwork::InputShapes getInputShape();

    std::string getInputName();

    std::string getOutputName();

    void createRequest(const cv::Mat &);

    void submitRequest(bool isAsync);

    //std::vector<float> doEverything(const cv::Mat&);
    cv::Mat doEverything(const cv::Mat&);

    const float* getResults();
};
