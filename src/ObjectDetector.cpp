#include "../include/ObjectDetector.hpp"
//#include <samples/ocv_common.hpp>

static InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat& mat)
{
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool is_dense =
        strideW == channels &&
        strideH == channels * width;

    if (!is_dense) THROW_IE_EXCEPTION
        << "Doesn't support conversion from not dense cv::Mat";

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
        { 1, channels, height, width },
        InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
};

Detector::Detector(const std::string &modelPath, const std::string &configPath, const InferenceEngine::Core& ie_) : modelPath(modelPath), configPath(configPath), ie(ie_)
{
    this->cnnNetwork = ie.ReadNetwork(modelPath, configPath);
    inputShapes = cnnNetwork.getInputShapes();
    this->inName = inputShapes.begin()->first;
    InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;

    inSizeVector[0] = 1;
    this->cnnNetwork.reshape(inputShapes);

    InferenceEngine::InputInfo& inputInfo = *(cnnNetwork).getInputsInfo().begin()->second;
    inputInfo.getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo.setLayout(InferenceEngine::Layout::NHWC);
    inputInfo.setPrecision(InferenceEngine::Precision::U8);

    const InferenceEngine::OutputsDataMap& outputsDataMap = cnnNetwork.getOutputsInfo();
    this->outName = outputsDataMap.begin()->first;
    InferenceEngine::Data& data = *outputsDataMap.begin()->second;

    const InferenceEngine::SizeVector& outSizeVector = data.getTensorDesc().getDims();
    this->out_blob_h = static_cast<int>(outSizeVector[2]);
    this->out_blob_w = static_cast<int>(outSizeVector[3]);

    this->executableNetwork = ie.LoadNetwork(this->cnnNetwork, "CPU");
}

void Detector::createRequest(const cv::Mat& image) {
    inferRequest = executableNetwork.CreateInferRequest();
    inferRequest.SetBlob(this->inName, wrapMat2Blob(image));
    inferRequest.Infer();
}


std::vector<DetectionObject> Detector::getDetections(const cv::Mat &image)
{
    std::vector<DetectionObject> detectedObjects;
    float width_  = static_cast<float>(image.cols);
    float height_ = static_cast<float>(image.rows);

    createRequest(image);

    InferenceEngine::LockedMemory<const void> outMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(inferRequest.GetBlob(this->outName))->rmap();
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
	if (score > 0.7)
	{
            DetectionObject det(classID, score, x0, y0, x1, y1);
            detectedObjects.push_back(det);
	}
    }
    return detectedObjects;
}
