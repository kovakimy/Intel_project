#include "../include/ReidNetwork.hpp"
//#include <samples/ocv_common.hpp>

#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

static InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat& mat)
{
	size_t channels = mat.channels();
	size_t height = mat.size().height;
	size_t width = mat.size().width;

	size_t strideH = mat.step.buf[0];
	size_t strideW = mat.step.buf[1];

	bool is_dense =
		(strideW == channels &&
		strideH == channels * width);

	if (!is_dense) THROW_IE_EXCEPTION
		<< "Doesn't support conversion from not dense cv::Mat";

	InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
		{ 1, channels, height, width },
		InferenceEngine::Layout::NHWC);

	return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
};


ReidentificationNet::ReidentificationNet(const string& _model_xml, const string& _model_bin,
	const Core& _ie) :
	modelXml(_model_xml), modelBin(_model_bin), ie(_ie)
{
	//2) Read a model IR created by the Model Optimizer (.xml is supported format):
	bin_network = ie.ReadNetwork(_model_xml, _model_bin);// , model_bin);

	//3) Configure input and output. Request input and output information using InferenceEngine::CNNNetwork::getInputsInfo(), 
	//and InferenceEngine::CNNNetwork::getOutputsInfo() methods:
	inputInfoReid = bin_network.getInputsInfo().begin()->second;
	outInfoReid = bin_network.getOutputsInfo().begin()->second;
	inputName = bin_network.getInputsInfo().begin()->first;
	outputName = bin_network.getOutputsInfo().begin()->first;
	inputShapes = bin_network.getInputShapes();

	//configure input
	inputInfoReid->setPrecision(Precision::U8);
	inputInfoReid->setLayout(Layout::NCHW);
	inputInfoReid->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
	inputInfoReid->getPreProcess().setColorFormat(ColorFormat::RGB);

	//configure otput
	outInfoReid->setPrecision(Precision::FP16);
	outInfoReid->setLayout(Layout::NC);

	//4) Load the model to the device using InferenceEngine::Core::LoadNetwork():
	executable_network = ie.LoadNetwork(bin_network, "CPU");

};

InputInfo::Ptr ReidentificationNet::getInput() {
	return inputInfoReid;
};

DataPtr ReidentificationNet::getOutput() {
	return outInfoReid;
};

string ReidentificationNet::getInputName() {
	return inputName;
};

string ReidentificationNet::getOutputName() {
	return outputName;
};

ICNNNetwork::InputShapes ReidentificationNet::getInputShape() {
	return inputShapes;
};

void ReidentificationNet::createRequest(const cv::Mat &pic_part){
	//5) Create an infer request
	request = executable_network.CreateInferRequest();
	//6) Prepare input
	//InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
	//	{ 1, (size_t)pic_part.channels(), (size_t)pic_part.size().height, (size_t)pic_part.size().width },
	//	InferenceEngine::Layout::NCHW);
	Blob::Ptr imgBlob = wrapMat2Blob(pic_part);//InferenceEngine::make_shared_blob<uint8_t>(tDesc, pic_part.data);
    request.SetBlob(inputName, imgBlob);

	width_ = static_cast<float>(pic_part.cols);
	height_ = static_cast<float>(pic_part.rows);
};

void ReidentificationNet::submitRequest(bool isAsync) {
	//7) Do inference 
	if (!isAsync) request.Infer();
	else {
		request.StartAsync();
		request.Wait(IInferRequest::WaitMode::RESULT_READY);
	};
}
const float* ReidentificationNet::getResults() {

   //8) Go over the output blobs and process the results.
	Blob::Ptr output = request.GetBlob(outputName);
	using myBlobType = PrecisionTrait<Precision::FP16>::value_type;
  //TBlob<myBlobType>& tblob = dynamic_cast<TBlob<myBlobType>&>(*output);
	auto const memLocker = output->cbuffer(); // use const memory locker
  //output_buffer is valid as long as the lifetime of memLocker
	const float* output_buffer = memLocker.as<const float*>();
	return output_buffer;
	};

//std::vector<float> ReidentificationNet::doEverything(const cv::Mat& picPart) {
//	this->createRequest(picPart);
//	this->submitRequest(false);
//	auto vectorData = this->getResults();
//	vector<float> resVector(256);
//	for (size_t i = 0; i < 256; ++i) {
//		resVector[i]=vectorData[i];
//	}
//	return resVector;
//};

cv::Mat ReidentificationNet::doEverything(const cv::Mat& picPart) {
	this->createRequest(picPart);
	this->submitRequest(false);
	auto vectorData = this->getResults();
	
	vector<float> resVector(256);
	for (size_t i = 0; i < 256; ++i) {
		resVector[i] = vectorData[i];
	}
	cv::Mat descr = cv::Mat(resVector);
	return descr;
}