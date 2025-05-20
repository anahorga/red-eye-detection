//
// Created by anaho on 5/4/2025.
//

#ifndef REDEYEDETECTION_EYE_H
#define REDEYEDETECTION_EYE_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef struct {
    Mat B;
    Mat G;
    Mat R;
} image_channels_bgr;
Mat eye_detection_haar_cascade(Mat source,vector<Rect> & haarCascadeEyes);
Rect findFaceRegion( Mat skinMask);
vector<Rect> findEyeCandidates( Mat mask,  Rect faceRect);
Mat detectSkin( Mat src);
vector<Rect> detectEyes( Mat img,  Rect faceRect);

Mat createRedEyeMask( Mat eye);

void fixRedEyes(Mat img, vector<Rect> eyes);
void correctRedEye(Mat eye, Mat mask);

void fillHoles(Mat mask);
void drawEyeCandidates(Mat image,  vector<Rect> candidates,  string windowName = "Eye Candidates");
double verifyPositionDifference(vector<Rect>selectedEyes,vector<Rect>myEyes);
void verifyColorDifference(Mat iphoneCorrect ,Mat myCorrect,vector<Rect>eyes);
#endif //REDEYEDETECTION_EYE_H
