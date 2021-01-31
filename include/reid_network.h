#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <ie_core.hpp>
//#include <inference_engine.hpp>

//model_reid = 'person-reidentification-retail-0286'

class Reidentification_results{};

class ReidentificationNet {
private:

    InferenceEngine::InferRequest request;
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork bin_network;
    InferenceEngine::ExecutableNetwork executable_network;

    std::string model_xml;
    std::string model_bin;

    InferenceEngine::InputInfo::Ptr input_info_reid;
    InferenceEngine::DataPtr  out_info_reid;
  //  InferenceEngine::InputsDataMap input_info_reid;
  //  InferenceEngine::OutputsDataMap  out_info_reid;
    std::string input_name;
    std::string output_name;
    InferenceEngine::ICNNNetwork::InputShapes input_shapes;

    float width_ = 0;
    float height_ = 0;

public:
    ReidentificationNet(const std::string&, const std::string&,
        const InferenceEngine::Core&);

    InferenceEngine::InputInfo::Ptr get_input();

    InferenceEngine::DataPtr get_output();

    InferenceEngine::ICNNNetwork::InputShapes get_input_shape();

    std::string get_input_name();

    std::string get_output_name();

    void createRequest(const cv::Mat &);

    void submitRequest(bool isAsync);

    InferenceEngine::Blob::Ptr getResults();
};
