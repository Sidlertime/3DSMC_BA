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
    Mat readPose(string path);
public:
    string name;
    bf_info info;
    int nImages;
    vector<Mat> imagesColor;
    vector<Mat> imagesDepth;
    vector<Mat> cameraPose;

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
    cameraPose = vector<Mat>();
}

DataloaderBF::~DataloaderBF(){
    imagesColor.clear();
    imagesDepth.clear();
    cameraPose.clear();
}

// Read the Set-Infos from the corresponding info.txt and save it in the info struct
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
    cout << "Sensor Name: " << info.sensorName << endl;
    infile.ignore(32, '=');
    infile >> info.colorWidth;
    cout << "Color Width: " << info.colorWidth << endl;
    infile.ignore(32, '=');
    infile >> info.colorHeight;
    cout << "Color Height: " << info.colorHeight << endl;
    infile.ignore(32, '=');
    infile >> info.depthWidth;
    cout << "Depth Width: " << info.depthWidth << endl;
    infile.ignore(32, '=');
    infile >> info.depthHeight;
    cout << "Depth Width: " << info.depthHeight << endl;
    infile.ignore(32, '=');
    infile >> info.depthShift;
    cout << "Depth Shift: " << info.depthShift << endl;

    // Color Intrinsic
    infile.ignore(32, '=');
    cout << "Color Intrinsic: ";
    for (int i = 0; i < 16; i++){
        infile >> info.calibrationColorIntrinsic[i];
        cout << info.calibrationColorIntrinsic[i] << " ";
    }
    cout << endl;
    // color Extrinsic
    infile.ignore(32, '=');
    cout << "Color Extrinsic: ";
    for (int i = 0; i < 16; i++){
        infile >> info.calibrationColorExtrinsic[i];
        cout << info.calibrationColorExtrinsic[i] << " ";
    }
    cout << endl;
    // Depth Intrinsic
    infile.ignore(32, '=');
    cout << "Depth Intrinsic: ";
    for (int i = 0; i < 16; i++){
        infile >> info.calibrationDepthIntrinsic[i];
        cout << info.calibrationDepthIntrinsic[i] << " ";
    }
    cout << endl;
    // Depth Extrinsic
    infile.ignore(32, '=');
    cout << "Depth Extrinsic: ";
    for (int i = 0; i < 16; i++){
        infile >> info.calibrationDepthExtrinsic[i];
        cout << info.calibrationDepthExtrinsic[i] << " ";
    }
    cout << endl;

    infile.ignore(32, '=');
    infile >> info.framesSize;
    cout << "Frame Size: " << info.framesSize << endl;

    infile.close();

    return true;
}

// Read the ground truth pose frame
Mat DataloaderBF::readPose(string path){
    ifstream infile(path);

    if(!infile.is_open()) {
        cout << "Failed to open file " << path << endl;
        return Mat::eye(4, 4, DataType<double>::type);
    }

    double *pose = new double[16];
    for (int i = 0; i < 16; i++){
        infile >> pose[i];
    }

    infile.close();
    
    return Mat(4, 4, DataType<double>::type, pose);
}

// return the corresponding frame name
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
        cameraPose.push_back(readPose(name + frame + POSE_SUFFIX));
    }

    cout << "Succesfully loaded set " << setName << " with " << nImages << " Images" << endl;

    return true;
}

