#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "eye.h"
#include <fstream>
using namespace std;
using namespace cv;

Mat bgr_2_YCrCb(Mat source) {
    int rows = source.rows, cols = source.cols;
    Mat dest(rows, cols, CV_8UC3); // output: YCrCb

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Vec3b pixel = source.at<Vec3b>(i, j);
            uchar B = pixel[0];
            uchar G = pixel[1];
            uchar R = pixel[2];

            //formule de conversie
            float fY  = 0.299f * R + 0.587f * G + 0.114f * B;
            float fCr = (R - fY) * 0.713f + 128;
            float fCb = (B - fY) * 0.564f + 128;

            uchar Y, Cr, Cb;

            if (fY < 0) Y = 0;
            else if (fY > 255) Y = 255;
            else Y = (uchar)fY;

            if (fCr < 0) Cr = 0;
            else if (fCr > 255) Cr = 255;
            else Cr = (uchar)fCr;

            if (fCb < 0) Cb = 0;
            else if (fCb > 255) Cb = 255;
            else Cb = (uchar)fCb;

            dest.at<Vec3b>(i, j) = Vec3b(Y, Cr, Cb);  // OpenCV: YCrCb order
        }
    }

    return dest;
}

Mat eye_detection_haar_cascade(Mat source, vector<Rect>& haarCascadeEyes)
{
    CascadeClassifier face_cascade("C:\\Users\\anaho\\OneDrive - Technical University of Cluj-Napoca\\Documents\\PI\\Proiect\\RedEyeDetection\\haarcascade_frontalface_default.xml");
    CascadeClassifier eye_cascade("C:\\Users\\anaho\\OneDrive - Technical University of Cluj-Napoca\\Documents\\PI\\Proiect\\RedEyeDetection\\haarcascade_eye.xml");

        Mat source_test = source.clone();
        Mat gray;
        cvtColor(source_test, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        GaussianBlur(gray, gray, Size(3, 3), 0); // reduce zgomotul

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(80, 80));

        for (const Rect& face : faces) {
            Mat faceROI = gray(face);
            vector<Rect> eyes;
            eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, 0, Size(15, 15));

            vector<Rect> validEyes;

            for (const Rect& eye : eyes) {
                int eyeCenterY = eye.y + eye.height / 2;
                if (eyeCenterY > face.height / 2) continue; // ignoră detecții sub mijlocul feței

                float aspect = (float)eye.width / eye.height;
                if (aspect < 0.5 || aspect > 2.5) continue; // ignoră forme ciudate

                validEyes.push_back(eye);
            }

            sort(validEyes.begin(), validEyes.end(), [](const Rect& a, const Rect& b) {
                return a.x < b.x; // sortare pe axa X: stânga → dreapta
            });

            if (validEyes.size() > 2)
                validEyes.resize(2); // doar primii 2 (cei mai probabili ochi)

            for (const Rect& eye : validEyes) {
                Rect eye_global(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
                rectangle(source_test, eye_global, Scalar(0, 255, 0), 2);
                haarCascadeEyes.push_back(eye_global);
            }
        }

        return source_test;


}

Mat manualInRange( Mat ycrcb) {
    int rows = ycrcb.rows;
    int cols = ycrcb.cols;
    Mat mask(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Vec3b pixel = ycrcb.at<Vec3b>(i, j);
            uchar Y  = pixel[0];
            uchar Cr = pixel[1];
            uchar Cb = pixel[2];

            // Interval tipic pentru piele
            if (Y >= 0 && Y <= 255 &&
                Cr >= 133 && Cr <= 173 &&
                Cb >= 77 && Cb <= 127) {
                mask.at<uchar>(i, j) = 255;
            } else {
                mask.at<uchar>(i, j) = 0;
            }
        }
    }

    return mask;
}
bool IsInside(Mat img, int i, int j){

    if(i>=img.rows||i<0||j<0||j>=img.cols)
        return false;

    return true;
}
Mat dilation( Mat source,int widthStr,int heightStr ,int no_iter) {
    Mat dst = source.clone();
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(widthStr, heightStr));
    int kCenter = kernel.rows / 2;

    for (int iter = 0; iter < no_iter; ++iter) {
        Mat aux = dst.clone();
        for (int i = 0; i < source.rows; ++i) {
            for (int j = 0; j < source.cols; ++j) {
                if (aux.at<uchar>(i, j) == 255) {
                    for (int ki = 0; ki < kernel.rows; ++ki) {
                        for (int kj = 0; kj < kernel.cols; ++kj) {
                            if (kernel.at<uchar>(ki, kj) > 0) {
                                int ni = i + ki - kCenter;
                                int nj = j + kj - kCenter;
                                if (IsInside(dst,ni,nj)) {
                                    dst.at<uchar>(ni, nj) = 255;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dst;
}

Mat erosion( Mat source,int widthStr,int heightStr, int no_iter) {
    Mat dst = source.clone();
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(widthStr, heightStr));
    int kCenter = kernel.rows / 2;

    for (int iter = 0; iter < no_iter; ++iter) {
        Mat aux = dst.clone();
        for (int i = 0; i < source.rows; ++i) {
            for (int j = 0; j < source.cols; ++j) {
                bool erodePixel = false;
                for (int ki = 0; ki < kernel.rows && !erodePixel; ++ki) {
                    for (int kj = 0; kj < kernel.cols; ++kj) {
                        if (kernel.at<uchar>(ki, kj) > 0) {
                            int ni = i + ki - kCenter;
                            int nj = j + kj - kCenter;
                            if (ni < 0 || ni >= dst.rows || nj < 0 || nj >= dst.cols || aux.at<uchar>(ni, nj) == 0) {
                                erodePixel = true;
                                break;
                            }
                        }
                    }
                }
                dst.at<uchar>(i, j) = erodePixel ? 0 : 255;
            }
        }
    }

    return dst;
}



Mat opening(Mat source,int widthStr,int heightStr ,int no_iter) {


    Mat dst=source.clone(), aux=source.clone();
    int rows=source.rows, cols=source.cols;


    for(int k=0;k<no_iter;k++)
    {
        dst = erosion(aux,widthStr,heightStr,1);
        dst = dilation(dst,widthStr,heightStr,1);
        aux = dst.clone();
    }


    return dst;

}

Mat closing(Mat source,int widthStr,int heightStr ,int no_iter) {


    Mat dst=source.clone(), aux=source.clone();
    int rows=source.rows, cols=source.cols;

    for(int k=0;k<no_iter;k++)
    {
        dst = dilation(aux,widthStr,heightStr,1);
        dst = erosion(dst,widthStr,heightStr,1);
        aux = dst.clone();
    }

    return dst;
}
Mat detectSkin( Mat src) {
    Mat ycrcb, mask;
    //cvtColor(src, ycrcb, COLOR_BGR2YCrCb); - functie open cv rescrisa de mine
    ycrcb=bgr_2_YCrCb(src);

    // Interval tipic pentru piele
    //inRange(ycrcb, Scalar(0, 133, 77), Scalar(255, 173, 127), mask); - functie openCV rescrisa de mine

    mask=manualInRange(ycrcb);
    // Curatare cu morfologie (dilatare urmata de eroziune)

    //morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    //morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    //mask=opening(mask,5,5,1);
    mask=closing(mask,5,5,1);

    return mask;
}

Rect findFaceRegion(Mat skinMask) {
    vector<vector<Point>> contours;

    //doar contururile exterioare si salvam punctele importante (de ex la o linie drepta salvam doar marginile)
    findContours(skinMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int maxArea = 0;
    Rect faceRect;

    for (const auto& contour : contours) {

        //dreptunghiul circumscris
        Rect r = boundingRect(contour);
        int area = r.area();

        //presupunem ca cea mai mare suprafata de piele este fata
        if (area > maxArea) {
            maxArea = area;
            faceRect = r;
        }
    }
    return faceRect;
}
Mat bgr_2_grayscale(Mat source){
    int rows=source.rows, cols=source.cols;
    Mat grayscale_image=Mat(rows,cols,CV_8UC1);

    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
        {
            grayscale_image.at<uchar>(i,j)=(source.at<Vec3b>(i,j)[0]+source.at<Vec3b>(i,j)[1]+source.at<Vec3b>(i,j)[2])/3;
        }

    return grayscale_image;
}
int otsuThreshold(int* hist, int total) {
    float sum = 0;
    for (int i = 0; i < 256; ++i)
        sum += i * hist[i];

    float sumB = 0;
    int wB = 0, wF = 0;

    float maxBetweenVar = 0;
    int threshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += hist[t]; // weight background
        if (wB == 0) continue;

        wF = total - wB; // weight foreground
        if (wF == 0) break;

        sumB += (float)(t * hist[t]);

        float mB = sumB / wB;           // mean background
        float mF = (sum - sumB) / wF;   // mean foreground

        float betweenVar = (float)wB * wF * (mB - mF) * (mB - mF);

        if (betweenVar > maxBetweenVar) {
            maxBetweenVar = betweenVar;
            threshold = t;
        }
    }

    return threshold;
}
int* compute_histogram_naive(Mat source){

    int* histogram = (int*)calloc(256, sizeof(int));

    int rows=source.rows,cols=source.cols;

    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
        {
            histogram[source.at<uchar>(i,j)]++;
        }

    return histogram;

}
Mat manualGaussianBlur( Mat src) {
    Mat dst = src.clone();

    // Kernel 5x5 gaussian aproximativ (sigma = 1.0)
    float kernel[5][5] = {
            { 1,  4,  6,  4, 1 },
            { 4, 16, 24, 16, 4 },
            { 6, 24, 36, 24, 6 },
            { 4, 16, 24, 16, 4 },
            { 1,  4,  6,  4, 1 }
    };

    float factor = 1.0f / 256.0f; // suma kernelului este 256

    for (int i = 2; i < src.rows - 2; ++i) {
        for (int j = 2; j < src.cols - 2; ++j) {
            float sum = 0.0f;
            for (int ki = -2; ki <= 2; ++ki) {
                for (int kj = -2; kj <= 2; ++kj) {
                    sum += kernel[ki + 2][kj + 2] * src.at<uchar>(i + ki, j + kj);
                }
            }
            dst.at<uchar>(i, j) = (uchar)(sum * factor);
        }
    }

    return dst;
}

Mat preprocessROI( Mat roi) {
    Mat gray, thresh;
    gray=bgr_2_grayscale(roi);
    //GaussianBlur(gray, gray, Size(5, 5), 0);
    //threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    gray=manualGaussianBlur(gray);
    int *histogram= compute_histogram_naive(gray);
    int threshold=otsuThreshold(histogram,gray.rows*gray.cols);
    thresh=gray.clone();
    for(int i=0;i<gray.rows;i++)
        for(int j=0;j<gray.cols;j++)
            if(gray.at<uchar>(i,j)>threshold)
                thresh.at<uchar>(i,j)=0;
            else
                thresh.at<uchar>(i,j)=255;
    //morphologyEx(thresh, thresh, MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    thresh=opening(thresh,3,3,1);

   // imshow("Masca fata",thresh);
    return thresh;

}

vector<Rect> findEyeCandidates( Mat mask,  Rect faceRect) {
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> candidates;

    for (const auto contour : contours) {
        Rect r = boundingRect(contour);

        //ajuta la aflarea unai forme rotunde
        float aspect = (float)r.width / r.height;

        int centerYCand = r.y + r.height / 2;
        int centerYFace = faceRect.y + faceRect.height / 2;

        //  filtru de pozitie verticala (doar sus)
        if (centerYCand > centerYFace + faceRect.height / 6)
            continue;

        //modifica niste parametrii
        if (r.width > 10 && r.width < 80 && //ignora forme foarte mici sau foarte mari
            r.height > 10 && r.height < 60 &&
            aspect > 0.7 && aspect < 3.0) //exclude forme ciudate
        {
            Rect eyeBox(r.x + faceRect.x, r.y + faceRect.y, r.width, r.height);
            candidates.push_back(eyeBox);
        }
    }

    return candidates;
}
void drawEyeCandidates(Mat image,  vector<Rect> candidates,  string windowName ) {
    for (size_t i = 0; i < candidates.size(); ++i) {
        const Rect r = candidates[i];
        rectangle(image, r, Scalar(0, 255, 255), 2);
        putText(image, to_string(i), Point(r.x, r.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }

    imshow(windowName, image);

}

pair<Rect, Rect> findBestEyePair( vector<Rect> candidates, int faceCenterX) {
    int bestScore = INT_MAX;
    pair<Rect, Rect> bestPair;

    //ponderi pentru scorul final
    int w_align = 3;
    int w_size = 2;
    int w_sym = 1; // mai mica pondere

    //luam toate perechile de candidati pentru ochi
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            Rect a = candidates[i], b = candidates[j];

            //aliniere orizonatala a ochilor
            int centerYDiff = abs((a.y + a.height / 2) - (b.y + b.height / 2));

            //ochii sa fie aproximativ egali ca marime
            int sizeDiff = abs(a.width - b.width) + abs(a.height - b.height);

            //ochii sunt pozitionati aproximativ simetric fata de centrul fetei
            int symA = abs((a.x + a.width / 2) - faceCenterX);
            int symB = abs((b.x + b.width / 2) - faceCenterX);
            int symDiff = abs(symA - symB);

            //distanta dintre ochi sa fie rezonabila
            int distX = abs((a.x + a.width / 2) - (b.x + b.width / 2));
            if (distX < 20 || distX > faceCenterX)  // elimina doar cazuri extreme
                continue;

            int totalScore = centerYDiff * w_align + sizeDiff * w_size + symDiff * w_sym;

            if (totalScore < bestScore) {
                bestScore = totalScore;
                bestPair = {a, b};
            }
        }
    }

    return bestPair;
}


vector<Rect> detectEyes( Mat img,  Rect faceRect) {
    Mat roi = img(faceRect).clone();
    Mat mask = preprocessROI(roi);
    vector<Rect> candidates = findEyeCandidates(mask, faceRect);
    Mat cand = img.clone();
    //drawEyeCandidates(cand,candidates);
    if (candidates.size() < 2)
        return {};

    pair<Rect, Rect> best = findBestEyePair(candidates, faceRect.x + faceRect.width / 2);

    // Validare (finala) daca sunt destul de aliniati si departati
    int deltaY = abs((best.first.y + best.first.height / 2) - (best.second.y + best.second.height / 2));
    int distX = abs((best.first.x + best.first.width / 2) - (best.second.x + best.second.width / 2));
    if (deltaY > faceRect.height / 5 || distX < 20 || distX > faceRect.width * 0.9)
        return {};
    if (best.first.x < best.second.x)
        return {best.first, best.second};
    else
        return {best.second, best.first};
}
image_channels_bgr break_channels(Mat source){

    int rows, cols;
    Mat B, G, R;
    image_channels_bgr bgr_channels;

    rows=source.rows;
    cols=source.cols;

    B=Mat(rows,cols,CV_8UC1);
    G=Mat(rows,cols,CV_8UC1);
    R=Mat(rows,cols,CV_8UC1);
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
        {
            B.at<uchar>(i,j)=source.at<Vec3b>(i,j)[0];

            G.at<uchar>(i,j)=source.at<Vec3b>(i,j)[1];

            R.at<uchar>(i,j)=source.at<Vec3b>(i,j)[2];


        }

    bgr_channels.B=B;
    bgr_channels.G=G;
    bgr_channels.R=R;
    return bgr_channels;
}


Mat createRedEyeMask( Mat eye) {
    image_channels_bgr bgr=break_channels(eye);

    Mat mask(bgr.R.rows, bgr.R.cols, CV_8UC1);

    for (int i = 0; i < bgr.R.rows; ++i) {
        for (int j = 0; j < bgr.R.cols; ++j) {
            uchar r = bgr.R.at<uchar>(i, j);
            uchar g = bgr.G.at<uchar>(i, j);
            uchar b = bgr.B.at<uchar>(i, j);

            // Heuristica: pixelul este considerat rosu daca:
            // rosul e mare (>150)
            // rosul este mai mare decat suma celorlalte doua canale
            if (r > 150 && r > (g + b)) {
                mask.at<uchar>(i, j) = 255;  // pixel marcat ca "ochi rosu"
            } else {
                mask.at<uchar>(i, j) = 0;    // pixel normal
            }
        }
    }

    return mask;
}

void fillHoles(Mat mask) {
    Mat mask_floodfill = mask.clone();
    floodFill(mask_floodfill, Point(0, 0), Scalar(255));

    Mat mask_inv;
    bitwise_not(mask_floodfill, mask_inv);

    mask = mask | mask_inv;
}
void correctRedEye(Mat eye,  Mat mask) {
    vector<Mat> bgr(3);
    split(eye, bgr);

    // Media canalelor verde si albastru
    Mat mean = (bgr[0] + bgr[1]) / 2;

    // Suprascriem toate cele 3 canale cu media
    mean.copyTo(bgr[0], mask);
    mean.copyTo(bgr[1], mask);
    mean.copyTo(bgr[2], mask);

    // Reconstruim imaginea ochiului
    merge(bgr, eye);
}
void fixRedEyes(Mat img,  vector<Rect> eyes) {
    for (const Rect eyeRect : eyes) {
        Mat eye = img(eyeRect);

        //   Creeaza masca
        Mat mask = createRedEyeMask(eye);

        //   Curata masca (umple gauri, dilateaza)
        fillHoles(mask);
        dilate(mask, mask, Mat(), Point(-1, -1), 3);
        //   Corecteaza ochiul
        correctRedEye(eye, mask);
    }
}
double computeIoU(Rect rectA, Rect rectB) {
    Rect intersection = rectA & rectB;
    double interArea = intersection.area();
    double unionArea = rectA.area() + rectB.area() - interArea;
    if (unionArea <= 0) return 0.0;
    return interArea / unionArea;
}
double verifyPositionDifference(vector<Rect>selectedEyes,vector<Rect>myEyes)
{
    if (selectedEyes.size() != 2 || myEyes.size() != 2) {
        cerr << " Nu sunt exact 2 ochi in fiecare vector" << endl;
        return 0.0;
    }
    double iou1 = computeIoU(selectedEyes[0], myEyes[0]);
    double iou2 = computeIoU(selectedEyes[1], myEyes[1]);
    double avgIoU = (iou1 + iou2) / 2.0;

    cout << "IoU ochi 1: " << iou1 << endl;
    cout << "IoU ochi 2: " << iou2 << endl;
    cout << "Media IoU: " << avgIoU << endl;

    return avgIoU;
}
void verifyColorDifference( Mat iphoneCorrect,  Mat myCorrect,  vector<Rect> eyes) {
    if (eyes.empty()) {
        cout << " Vectorul de ochi este gol" << endl;
        return;
    }

    double totalDiff = 0.0;
    int totalPixels = 0;

    for ( Rect eye : eyes) {
        Rect validEye = eye & Rect(0, 0, iphoneCorrect.cols, iphoneCorrect.rows); // clamp
        for (int y = validEye.y; y < validEye.y + validEye.height; ++y) {
            for (int x = validEye.x; x < validEye.x + validEye.width; ++x) {
                Vec3b pixIphone = iphoneCorrect.at<Vec3b>(y, x);
                Vec3b pixMine   = myCorrect.at<Vec3b>(y, x);

                // diferenta absoluta pe fiecare canal
                int db = abs(pixIphone[0] - pixMine[0]);
                int dg = abs(pixIphone[1] - pixMine[1]);
                int dr = abs(pixIphone[2] - pixMine[2]);

                totalDiff += (db + dg + dr) / 3.0;
                totalPixels++;
            }
        }
    }

    if (totalPixels == 0) {
        cout << " Nu s au gasit pixeli valizi in dreptunghiurile ochilor" << endl;
        return;
    }

    double avgDiff = totalDiff / totalPixels;
    cout << "Diferenta medie de culoare pe pixel in zona ochilor: " << avgDiff << " (din 255)" << endl;
}



