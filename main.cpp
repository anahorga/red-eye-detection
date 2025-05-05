#include <iostream>

#include <opencv2/opencv.hpp>
#include "src/eye.h"
using namespace std;
using namespace cv;


int main() {


    //Mat source = imread("C:\\Users\\anaho\\OneDrive - Technical University of Cluj-Napoca\\Documents\\PI\\Proiect\\RedEyeDetection\\images\\redeyes1.jpg",
                  //IMREAD_COLOR);

    Mat source = imread("C:\\Users\\anaho\\OneDrive - Technical University of Cluj-Napoca\\Documents\\PI\\Proiect\\RedEyeDetection\\images\\img4.jpg",
                    IMREAD_COLOR);

    imshow("Original Image", source);


    Mat skinMask = detectSkin(source);
    Rect face = findFaceRegion(skinMask);


    Mat eyesRect=source.clone();
    vector<Rect> eyes = detectEyes(source, face);
    for (const Rect& eye : eyes) {
        rectangle(eyesRect, eye, Scalar(255, 0, 0), 2); // contur ochi
    }
    imshow("Ochi detectati", eyesRect);


    // Aplicați corectarea
    fixRedEyes(source, eyes);

    imshow("Ochi corectati", source);

    waitKey(0);
    return 0;
}
