#include "../include/detector.hpp"
#include <samples/ocv_common.hpp>

Detector::Detector(std::string &modelPath, std::string &configPath) : modelPath(modelPath), configPath(configPath)
{
	this->cnnNetwork = ie.ReadNetwork(modelPath, configPath);
}

std::vector<DetectionObject> Detector::getDetections(cv::Mat &image)
{
    std::vector<DetectionObject> detectedObjects;
    float width_  = static_cast<float>(image.cols);
    float height_ = static_cast<float>(image.rows);

    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    const std::string& inName = inputShapes.begin()->first;
    InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;

    inSizeVector[0] = 1;
    this->cnnNetwork.reshape(inputShapes);

    InferenceEngine::InputInfo& inputInfo = *(cnnNetwork).getInputsInfo().begin()->second;
    inputInfo.getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo.setLayout(InferenceEngine::Layout::NHWC);
    inputInfo.setPrecision(InferenceEngine::Precision::U8);

    const InferenceEngine::OutputsDataMap& outputsDataMap = cnnNetwork.getOutputsInfo();
    const std::string& outName = outputsDataMap.begin()->first;
    InferenceEngine::Data& data = *outputsDataMap.begin()->second;

    const InferenceEngine::SizeVector& outSizeVector = data.getTensorDesc().getDims();
    const int out_blob_h = static_cast<int>(outSizeVector[2]);
    const int out_blob_w = static_cast<int>(outSizeVector[3]);

    InferenceEngine::ExecutableNetwork executableNetwork = ie.LoadNetwork(this->cnnNetwork, "CPU");
    InferenceEngine::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    inferRequest.SetBlob(inName, wrapMat2Blob(image));
    inferRequest.Infer();

    InferenceEngine::LockedMemory<const void> outMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(inferRequest.GetBlob(outName))->rmap();
    const float *data_ = outMapped.as<float *>();

    for (int det_id = 0; det_id < out_blob_h; ++det_id)
    {
        const int start_pos = det_id * out_blob_w;
        const int imageID   = data_[start_pos];
        const int classID   = data_[start_pos + 1];
        const double score  = std::min(std::max(0.0f, data_[start_pos + 2]), 1.0f);
        const double x0     = std::min(std::max(0.0f, data_[start_pos + 3]), 1.0f) * width_;
        const double y0     = std::min(std::max(0.0f, data_[start_pos + 4]), 1.0f) * height_;
        const double x1     = std::min(std::max(0.0f, data_[start_pos + 5]), 1.0f) * width_;
        const double y1     = std::min(std::max(0.0f, data_[start_pos + 6]), 1.0f) * height_;
        DetectionObject det(classID, score, x0, y0, x1, y1);
        detectedObjects.push_back(det);
    }
    return detectedObjects;
}