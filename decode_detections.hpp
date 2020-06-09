#ifndef DECODE_DETECTIONS_HPP__
#define DECODE_DETECTIONS_HPP__ 

#include <vector>
#include <Eigen/Dense>
// #include <opencv2/opencv.hpp>
#include <iostream>
// #include <chrono> 
// #include <fstream>
// #include <numeric>


using namespace Eigen;
using namespace std;
// using namespace std::chrono;
using std::vector;
// using cv::Rect;
// using cv::Point;


MatrixXf decode_detections(const MatrixXf & y_pred, const float & confidence_thresh=0.3, const float & iou_threshold=0.45, const int & top_k=200, const int & img_height=300, const int & img_width=300);

MatrixXf convert_coordinates(const MatrixXf & matrix);

MatrixXf vectorized_nms(const MatrixXf & boxes, const float & iou_thresh);

VectorXi argsort_eigen(VectorXf & vec);

void append_int_eigen(VectorXi & vect, int & value);

VectorXf extract_values(VectorXf & vec, VectorXi & idxs);

VectorXf max_eigen(VectorXf & vec1, int & i, VectorXf & vec2);

VectorXf min_eigen(VectorXf & vec1, int & i, VectorXf & vec2);

#endif // DECODE_DETECTIONS_HPP__ 