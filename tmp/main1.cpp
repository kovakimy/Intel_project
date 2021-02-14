#include <string>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
//#include <inference_engine.hpp>
//#include <ie_core.hpp>
#include <inference_engine.hpp>

#include <iostream>
#include "include/reid_network.h"


using namespace std;
using namespace cv;

int main() {
	std::string image_path = "F:/obj_det_prj/test/ped1.png";
	Mat img = imread(image_path, IMREAD_COLOR);
	if (img.empty())
	{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return 1;
	}

;
	Mat res;


	//const char model_reid1[] = "model_name.xml";
	//const char model_reid2[] = "model_name.bin";
	const std::string s1 = "person-reidentification-retail-0286.xml";
	const std::string s2 = "person-reidentification-retail-0286.bin";
	const std::string s3 = "F:/obj_det_prj/person-reidentification-retail-0286/FP16/";
	const std::string s4 = "";
	//InferenceEngine::Core ie;
	

	//auto re_id_model = ie.ReadNetwork(model_reid1, model_reid2);
	// 1) Create Inference Engine Core to manage available devices and read network objects:
	InferenceEngine::Core ie;
	//InferenceEngine::CNNNetwork bin_network =_ie.ReadNetwork(s3+s1, s3+s2);
	ReidentificationNet ri(s3 + s1, s3 + s2, ie);

	auto nnn = ri.getInputShape();

	ri.createRequest(img);
	ri.submitRequest(false);
	// const float* ress1 = new float[256];
	// ress1 = ri.getResults();
	float* ress1 = (float*) calloc(256, sizeof(float));
	memcpy(ress1, ri.getResults(), 256);

	image_path = "F:/obj_det_prj/test/ped5.png";
	img = imread(image_path, IMREAD_COLOR);

	ri.createRequest(img);
	ri.submitRequest(false);
	float* ress2 = (float*)calloc(256, sizeof(float));
	memcpy(ress2, ri.getResults(), 256);

	cout << cosineSimilarity(ress1, ress2, 256) << endl;

	cout << "test";//
	
	return 0;
}