#include "F:\obj_det_prj\include\reid_network.h"

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


ReidentificationNet::ReidentificationNet(const string& _model_xml, const string& _model_bin,
	const Core& _ie) :
	model_xml(_model_xml), model_bin(_model_bin), ie(_ie)
{
	//2) Read a model IR created by the Model Optimizer (.xml is supported format):
	bin_network = ie.ReadNetwork(_model_xml, _model_bin);// , model_bin);

	//3) Configure input and output. Request input and output information using InferenceEngine::CNNNetwork::getInputsInfo(), 
	//and InferenceEngine::CNNNetwork::getOutputsInfo() methods:
	input_info_reid = bin_network.getInputsInfo().begin()->second;
	out_info_reid = bin_network.getOutputsInfo().begin()->second;
	input_name = bin_network.getInputsInfo().begin()->first;
	output_name = bin_network.getInputsInfo().begin()->first;
	input_shapes = bin_network.getInputShapes();

	//configure input
	input_info_reid->setPrecision(InferenceEngine::Precision::U8);
	input_info_reid->setLayout(InferenceEngine::Layout::NCHW);
	input_info_reid->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
	input_info_reid->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);

	//configure otput
	out_info_reid->setPrecision(InferenceEngine::Precision::FP16);
	out_info_reid->setLayout(InferenceEngine::Layout::NC);

	//4) Load the model to the device using InferenceEngine::Core::LoadNetwork():
	executable_network = ie.LoadNetwork(bin_network, "CPU");

};

InputInfo::Ptr ReidentificationNet::get_input() {
	return input_info_reid;
};

DataPtr ReidentificationNet::get_output() {
	return out_info_reid;
};

string ReidentificationNet::get_input_name() {
	return input_name;
};

string ReidentificationNet::get_output_name() {
	return output_name;
};

ICNNNetwork::InputShapes ReidentificationNet::get_input_shape() {
	return input_shapes;
};

void ReidentificationNet::createRequest(const cv::Mat &pic_part){
	//5) Create an infer request
	request = executable_network.CreateInferRequest();
	/*
	6) Prepare input
	*/
	Blob::Ptr imgBlob = wrapMat2Blob(pic_part);
    request.SetBlob(input_name, imgBlob);

	width_ = static_cast<float>(pic_part.cols);
	height_ = static_cast<float>(pic_part.rows);
};

void ReidentificationNet::submitRequest(bool isAsync) {
	//7) Do inference 
	if (!isAsync) request.Infer();
	else {
		request.StartAsync();
		request.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
	};
}
Blob::Ptr ReidentificationNet::getResults() {
		//8) Go over the output blobs and process the results.
		Blob::Ptr output = request.GetBlob(output_name);
		//results
	//	return output;
		using myBlobType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type;
	//	TBlob<myBlobType>& tblob = dynamic_cast<TBlob<myBlobType>&>(*output);
	    auto const memLocker = output->cbuffer(); // use const memory locker
	 // output_buffer is valid as long as the lifetime of memLocker
		const float* output_buffer = memLocker.as<const float*>();
		return output;

	};
