#include <iostream>

#include <opencv2/opencv.hpp>
#include "src/eye.h"
using namespace std;
using namespace cv;

vector<Rect> selectedEyes;
vector<Rect> haarCascadeEyes;
Point startPoint;
bool drawing = false;
Mat image, tempImage;

void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        drawing = true;
        startPoint = Point(x, y);
    } else if (event == EVENT_MOUSEMOVE && drawing) {
        tempImage = image.clone();
        rectangle(tempImage, startPoint, Point(x, y), Scalar(0, 255, 0), 2);
        imshow("Selecteaza ochii", tempImage);
    } else if (event == EVENT_LBUTTONUP && drawing) {
        drawing = false;
        Rect selectedRect = Rect(startPoint, Point(x, y));
        selectedRect = selectedRect & Rect(0, 0, image.cols, image.rows); // clamp

        if (selectedRect.area() > 0 && selectedEyes.size() < 2) {
            selectedEyes.push_back(selectedRect);
            rectangle(image, selectedRect, Scalar(0, 255, 0), 2);
            imshow("Selecteaza ochii", image);
        }
    }
}


int main() {


    //Mat source = imread("C:\\Users\\anaho\\OneDrive - Technical University of Cluj-Napoca\\Documents\\PI\\Proiect\\RedEyeDetection\\images\\redeyes1.jpg",
          //       IMREAD_COLOR);

    Mat source = imread("C:\\Users\\anaho\\OneDrive - Technical University of Cluj-Napoca\\Documents\\PI\\Proiect\\RedEyeDetection\\images\\img1.png",
                   IMREAD_COLOR);



    imshow("Original Image", source);


    //ochide selectati de mana
    image=source.clone();
    tempImage = image.clone();
    namedWindow("Selecteaza ochii");
    setMouseCallback("Selecteaza ochii", onMouse);

    imshow("Selecteaza ochii", image);
    cout << "Selecteaza manual 2 ochi. Apasa orice tasta cand ai terminat.\n";

    waitKey(0); // dupa selectare

    //imshow("Ochi Selectati manual",image);



     //ochi selectati folosind haar cascade
    Mat source_test=eye_detection_haar_cascade(source,haarCascadeEyes);
    imshow("ochi detectati cu haar cascade",source_test);


    //detectia si corectia ochilor automata (programul meu)
    Mat skinMask = detectSkin(source);

    Rect face = findFaceRegion(skinMask);



    Mat eyesRect=source.clone();
    vector<Rect> eyes = detectEyes(source, face);
    for (const Rect& eye : eyes) {
        rectangle(eyesRect, eye, Scalar(255, 0, 0), 2); // contur ochi
        rectangle(image, eye, Scalar(255, 0, 0), 2); // contur ochi
        rectangle(source_test, eye, Scalar(255, 0, 0), 2); // contur ochi

    }
    imshow("Ochi detectati automat cu programul meu", eyesRect);
    imshow("Selectie Manuala VS Selectie automata ", image);
    imshow("Selectie HaarCascade VS Selectie automata ", source_test);


    fixRedEyes(source, eyes);

    //fixRedEyes(source,selectedEyes);

    imshow("Ochi corectati", source);

    Mat correct_iphone = imread("C:\\Users\\anaho\\OneDrive - Technical University of Cluj-Napoca\\Documents\\PI\\Proiect\\RedEyeDetection\\images\\pozaCorectieOchiRosiiIphone-img1.jpg",
                        IMREAD_COLOR);
    imshow("Ochi corectati cu functia din iphone",correct_iphone);



    cout<<"Comparatie detectie manuala vs detectie automata"<<endl;
    verifyPositionDifference(selectedEyes,eyes);//iou(intersection over union) cu selectie manuala

    cout <<endl;

   cout<<"Comparatie detectie haar cascade vs detectie automata"<<endl;
    verifyPositionDifference(haarCascadeEyes,eyes);//iou(intersection over union) cu haar cascade
    verifyColorDifference(correct_iphone,source,eyes);//verificam
    //verifyColorDifference(correct_iphone,source,selectedEyes);
    //culoarea pixelilor corectati de programul meu si cei corectati de o functie de standard industrail(cea din iphone)



    waitKey(0);
    return 0;
}
