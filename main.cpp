#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main() {


    Mat source = imread("C:\\Users\\anaho\\OneDrive - Technical University of Cluj-Napoca\\Documents\\PI\\Proiect\\RedEyeDetection\\images\\redeyes1.jpg",
                        IMREAD_COLOR);

    imshow("Original Image", source);

    waitKey(0);
    return 0;
}
