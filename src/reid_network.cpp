#include "F:\obj_det_prj\include\reid_network.h"
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
		strideW == channels &&
		strideH == channels * width;

	if (!is_dense) THROW_IE_EXCEPTION
		<< "Doesn't support conversion from not dense cv::Mat";

	InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
		{ 1, channels, height, width },
		InferenceEngine::Layout::NHWC);

	return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
};

float cosineSimilarity(const float* A, const float* B, size_t size) {
	float res = 0;
	float sumA2 = 0;
	float sumB2 = 0;
	float sumAB = 0;
	for (size_t i = 0; i < size; ++i) {
		sumAB += A[i] * B[i];
		sumA2 += A[i] * A[i];
		sumB2 += B[i] * B[i];
	}
	res = sumAB / (sqrt(sumA2) * sqrt(sumB2));
	return res;
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
	/*
	6) Prepare input
	*/
	Blob::Ptr imgBlob = wrapMat2Blob(pic_part);
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
	//	TBlob<myBlobType>& tblob = dynamic_cast<TBlob<myBlobType>&>(*output);
	    auto const memLocker = output->cbuffer(); // use const memory locker
	 // output_buffer is valid as long as the lifetime of memLocker
		const float* output_buffer = memLocker.as<const float*>();
		const float* A = output_buffer;
		return output_buffer;

	};
