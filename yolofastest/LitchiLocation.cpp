#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"
#include<iostream>

#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include<string>
namespace py = pybind11;
// 这个是可以抽象出来的
class LitchiLocation
{
	public:
		LitchiLocation(std::string& param_path, std::string& model_path);
		py::list locate(py::array_t<unsigned char>& input);
	private:
	//ndarray->mat
		cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input);
		ncnn::Net detector;
		int detector_size_width;
		int detector_size_height;
};

LitchiLocation::LitchiLocation(std::string& param_path, std::string& model_path)
{
	this->detector.opt.use_packing_layout = true;
    this->detector.opt.use_bf16_storage = true;  
    this->detector.load_param((char*)param_path.c_str());
    this->detector.load_model((char*)model_path.c_str());
	this->detector_size_height = 320;
	this->detector_size_width = 320;
}

py::list LitchiLocation::locate(py::array_t<unsigned char>& input)
{        
	//static const char* class_names[] = {"background","pupil"};
	cv::Mat image = this->numpy_uint8_3c_to_cv_mat(input);
	cv::Mat bgr = image.clone();
	int img_w = bgr.cols;
    int img_h = bgr.rows;
	py::list res;
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                 bgr.cols, bgr.rows, this->detector_size_width, this->detector_size_height);

	    //数据预处理
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
	in.substract_mean_normalize(mean_vals, norm_vals);
	
	ncnn::Extractor ex = detector.create_extractor();
    ex.set_num_threads(4);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);
	
	for (int i = 0; i < out.h; i++)
    {
        int label;
        float x1, y1, x2, y2, score;
        const float* values = out.row(i);
        
        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;

        score = values[1];
        label = values[0];

        //处理坐标越界问题
        if(x1<0) x1=0;
        if(y1<0) y1=0;
        if(x2<0) x2=0;
        if(y2<0) y2=0;

        if(x1>img_w) x1=img_w;
        if(y1>img_h) y1=img_h;
        if(x2>img_w) x2=img_w;
        if(y2>img_h) y2=img_h;
		
		res.append(label);
		res.append(x1);
		res.append(y1);
		res.append(x2);
		res.append(y2);
		res.append(score);
    }
    return res;
}
cv::Mat LitchiLocation::numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input) {

    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");

    py::buffer_info buf = input.request();

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    return mat;
}

PYBIND11_MODULE(LitchiLocation, m) {
    
    py::class_<LitchiLocation>(m, "LitchiLocation")
        .def(py::init<std::string&, std::string&>())
        .def("locate", &LitchiLocation::locate);

}
