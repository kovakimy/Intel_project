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

	std::string image_path = "F:/obj_det_prj/997.png";
	Mat img = imread(image_path, IMREAD_COLOR);
	if (img.empty())
	{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return 1;
	}
	//img.resize(100);
	imshow("Display window", img);
	Mat res;
//	cv::Mat::resize(img, res, cv::Size(100,100));
	//img.resize(100);
	//int k = waitKey(0); // Wait for a keystroke in the window
//	if (k == 's')
//	{
//		imwrite("starry_night.png", img);
//	}

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

	auto nnn = ri.get_input_shape();

	ri.createRequest(img);
	ri.submitRequest(false);
	auto ress = ri.getResults();

	cout << "test";// 
	return 0;
}