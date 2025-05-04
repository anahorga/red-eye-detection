//
// Created by anaho on 5/4/2025.
//

#ifndef REDEYEDETECTION_EYE_H
#define REDEYEDETECTION_EYE_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


Rect findFaceRegion(const Mat& skinMask);

Mat detectSkin(const Mat& src);
vector<Rect> detectEyes(const Mat& img, const Rect& faceRect);

Mat createRedEyeMask(const Mat& eye);

void fixRedEyes(Mat& img, const vector<Rect>& eyes);
void correctRedEye(Mat& eye, const Mat& mask);

void fillHoles(Mat& mask);

#endif //REDEYEDETECTION_EYE_H
