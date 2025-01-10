#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define BF_PATH "../Data/Bundle Fusion/"
#define FRAME_PREFIX "/frame-"
#define COLOR_SUFFIX ".color.jpg"
#define DEPTH_SUFFIX ".depth.png"
#define POSE_SUFFIX ".pose.txt"

using namespace std;
using namespace cv;

struct bf_info {
    int versionNumber;
    string sensorName;
    int colorWidth;
    int colorHeight;
    int depthWidth;
    int depthHeight;
    int depthShift;
    double calibrationColorIntrinsic[16];
    double calibrationColorExtrinsic[16];
    double calibrationDepthIntrinsic[16];
    double calibrationDepthExtrinsic[16];
    int framesSize;
};

class DataloaderBF
{
private:
    bool readInfo(string path);
    string toFrameName(int number);
public:
    string name;
    bf_info info;
    int nImages;
    vector<Mat> imagesColor;
    vector<Mat> imagesDepth;
    vector<Mat4d> cameraPose;

    DataloaderBF();
    ~DataloaderBF();

    bool loadImages(string name, int number);
};

DataloaderBF::DataloaderBF(){
    name = "";
    info = {};
    nImages = 0;
    imagesColor = vector<Mat>();
    imagesDepth = vector<Mat>();
    cameraPose = vector<Mat4d>();
}

DataloaderBF::~DataloaderBF(){
}

bool DataloaderBF::readInfo(string path){
    ifstream infile(path);

    if(!infile.is_open()) {
        cout << "Failed to open file " << path << endl;
        return false;
    }



    infile.ignore(32, '=');
    infile >> info.versionNumber;
    cout << "Version Number: " << info.versionNumber << endl;
    infile.ignore(32, '=');
    infile.ignore(1);
    getline(infile, info.sensorName);
    infile.ignore(32, '=');
    infile >> info.colorWidth;
    infile.ignore(32, '=');
    infile >> info.colorHeight;
    infile.ignore(32, '=');
    infile >> info.depthWidth;
    infile.ignore(32, '=');
    infile >> info.depthHeight;
    infile.ignore(32, '=');
    infile >> info.depthShift;

    // Color Intrinsic
    infile.ignore(32, '=');
    for (int i = 0; i<16; i++){
        infile >> info.calibrationColorIntrinsic[i];
    }
    // color Extrinsic
    infile.ignore(32, '=');
    for (int i = 0; i<16; i++){
        infile >> info.calibrationColorExtrinsic[i];
    }
    // Depth Intrinsic
    infile.ignore(32, '=');
    for (int i = 0; i<16; i++){
        infile >> info.calibrationDepthIntrinsic[i];
    }
    // Depth Extrinsic
    infile.ignore(32, '=');
    for (int i = 0; i<16; i++){
        infile >> info.calibrationDepthExtrinsic[i];
    }

    infile.ignore(32, '=');
    infile >> info.framesSize;

    infile.close();

    return true;
}

string DataloaderBF::toFrameName(int number){
    stringstream ss;
    ss << setw(6) << setfill('0') << number;
    return FRAME_PREFIX + ss.str();
}

bool DataloaderBF::loadImages(string setName, int number = -1){
    name = (string)BF_PATH + setName;

    // Read info.txt file
    if(!readInfo(name + "/info.txt")){
        /* error while reading info file */
        return false;
    }

    if(number < 0) number = info.framesSize;
    nImages = min(number, info.framesSize);

    // read images
    for (int i = 0; i < nImages; i++){
        string frame = toFrameName(i);
        imagesColor.push_back(imread(name + frame + COLOR_SUFFIX));
        imagesDepth.push_back(imread(name + frame + DEPTH_SUFFIX));
        //cameraPose.push_back()
    }

    cout << "Succesfully loaded set " << setName << " with " << nImages << " Images" << endl;

    return true;
}

