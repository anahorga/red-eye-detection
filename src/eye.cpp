//
// Created by anaho on 5/4/2025.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "eye.h"
#include <fstream>
using namespace std;
using namespace cv;

Mat detectSkin(const Mat& src) {
    Mat ycrcb, mask;
    cvtColor(src, ycrcb, COLOR_BGR2YCrCb);

    // Interval tipic pentru piele
    inRange(ycrcb, Scalar(0, 133, 77), Scalar(255, 173, 127), mask);

    // Curatare cu morfologie (dilatare urmata de eroziune)
    morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    return mask;
}

Rect findFaceRegion(const Mat& skinMask) {
    vector<vector<Point>> contours;
    findContours(skinMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int maxArea = 0;
    Rect faceRect;

    for (const auto& contour : contours) {
        Rect r = boundingRect(contour);
        int area = r.area();
        if (area > maxArea) {
            maxArea = area;
            faceRect = r;
        }
    }
    return faceRect;
}

Mat preprocessROI(const Mat& roi) {
    Mat gray, thresh;
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    morphologyEx(thresh, thresh, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    imshow("Masca fata",thresh);
    return thresh;

}

vector<Rect> findEyeCandidates(const Mat& mask, const Rect& faceRect) {
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> candidates;

    for (const auto& contour : contours) {
        Rect r = boundingRect(contour);
        float aspect = (float)r.width / r.height;

        int centerYCand = r.y + r.height / 2;
        int centerYFace = faceRect.y + faceRect.height / 2;

        // adăugăm filtru de poziție verticală (doar sus)
        if (centerYCand > centerYFace + faceRect.height / 6)
            continue;

        //modifica niste parametrii
        if (r.width > 10 && r.width < 80 &&
            r.height > 10 && r.height < 60 &&
            aspect > 0.7 && aspect < 3.0) {
            Rect eyeBox(r.x + faceRect.x, r.y + faceRect.y, r.width, r.height);
            candidates.push_back(eyeBox);
        }
    }

    return candidates;
}
void drawEyeCandidates(Mat& image, const vector<Rect>& candidates, const string& windowName ) {
    for (size_t i = 0; i < candidates.size(); ++i) {
        const Rect& r = candidates[i];
        rectangle(image, r, Scalar(0, 255, 255), 2);
        putText(image, to_string(i), Point(r.x, r.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }

    imshow(windowName, image);

}

pair<Rect, Rect> findBestEyePair(const vector<Rect>& candidates, int faceCenterX) {
    int bestScore = INT_MAX;
    pair<Rect, Rect> bestPair;

    int w_align = 3;
    int w_size = 2;
    int w_sym = 1; // mai mica pondere

    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            Rect a = candidates[i], b = candidates[j];

            int centerYDiff = abs((a.y + a.height / 2) - (b.y + b.height / 2));
            int sizeDiff = abs(a.width - b.width) + abs(a.height - b.height);
            int symA = abs((a.x + a.width / 2) - faceCenterX);
            int symB = abs((b.x + b.width / 2) - faceCenterX);
            int symDiff = abs(symA - symB);
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


vector<Rect> detectEyes(const Mat& img, const Rect& faceRect) {
    Mat roi = img(faceRect).clone();
    Mat mask = preprocessROI(roi);
    vector<Rect> candidates = findEyeCandidates(mask, faceRect);
    Mat cand = img.clone();
    drawEyeCandidates(cand,candidates);
    if (candidates.size() < 2)
        return {};

    pair<Rect, Rect> best = findBestEyePair(candidates, faceRect.x + faceRect.width / 2);

    // Validam dacă sunt destul de aliniati si simetrici
    int deltaY = abs((best.first.y + best.first.height / 2) - (best.second.y + best.second.height / 2));
    int distX = abs((best.first.x + best.first.width / 2) - (best.second.x + best.second.width / 2));
    if (deltaY > faceRect.height / 5 || distX < 20 || distX > faceRect.width * 0.9)
        return {};
    return {best.first, best.second};
}

Mat createRedEyeMask(const Mat& eye) {
    vector<Mat> bgr(3);
    split(eye, bgr);

    // Heuristica: rosu > 150 si rosu > verde + albastru
    // se poate adapta in functie de poza
    Mat mask = (bgr[2] > 150) & (bgr[2] > (bgr[1] + bgr[0]));

    // Convertim la uint8 (0 sau 255)
    mask.convertTo(mask, CV_8U, 255);

    return mask;
}

void fillHoles(Mat& mask) {
    Mat mask_floodfill = mask.clone();
    floodFill(mask_floodfill, Point(0, 0), Scalar(255));

    Mat mask_inv;
    bitwise_not(mask_floodfill, mask_inv);

    mask = mask | mask_inv;
}
void correctRedEye(Mat& eye, const Mat& mask) {
    vector<Mat> bgr(3);
    split(eye, bgr);

    // Media canalelor verde și albastru
    Mat mean = (bgr[0] + bgr[1]) / 2;

    // Suprascriem toate cele 3 canale cu media
    mean.copyTo(bgr[0], mask);
    mean.copyTo(bgr[1], mask);
    mean.copyTo(bgr[2], mask);

    // Reconstruim imaginea ochiului
    merge(bgr, eye);
}
void fixRedEyes(Mat& img, const vector<Rect>& eyes) {
    for (const Rect& eyeRect : eyes) {
        Mat eye = img(eyeRect);

        // Pasul 1: Creează mască
        Mat mask = createRedEyeMask(eye);

        // Pasul 2: Curăță masca (umple găuri, dilatează)
        fillHoles(mask);
        dilate(mask, mask, Mat(), Point(-1, -1), 3);

        // Pasul 3: Corectează ochiul
        correctRedEye(eye, mask);
    }
}




