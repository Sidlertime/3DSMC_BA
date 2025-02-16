#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using std::vector;
using cv::DMatch;
using cv::KeyPoint;
using cv::Mat;

class ComparisonPair{
public:
    ComparisonPair(): cam_idx_1(-1), cam_idx_2(-1){
        init_matrices();
    }
    ComparisonPair(const int cam1, const int cam2, const vector<DMatch>& matches, 
        const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2): 
        cam_idx_1(cam1), cam_idx_2(cam2), matches(matches), keypoints_1(keypoints1), keypoints_2(keypoints2){
        init_matrices();
    }
    ~ComparisonPair(){
        matches.clear();
        keypoints_1.clear();
        keypoints_2.clear();
    }

    void init_matrices(){
        intrinsics_mat_1 = Mat::eye(3, 3, CV_32F);
        intrinsics_mat_2 = Mat::eye(3, 3, CV_32F);
        extrinsics_mat_1 = Mat::eye(3, 4, CV_32F);
        extrinsics_mat_2 = Mat::eye(3, 4, CV_32F);
    }

    void update_intrinsics(float fx, float fy, float cx, float cy){
        intrinsics_mat_1.at<float>(0, 0) = fx;
        intrinsics_mat_1.at<float>(1, 1) = fy;
        intrinsics_mat_1.at<float>(0, 2) = cx;
        intrinsics_mat_1.at<float>(1, 2) = cy;
        intrinsics_mat_1.at<float>(2, 2) = 1;
        intrinsics_mat_1.copyTo(intrinsics_mat_2);
    }

    int cam_idx_1;
    int cam_idx_2;
    vector<DMatch> matches;
    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    Mat intrinsics_mat_1;
    Mat intrinsics_mat_2;
    Mat F;
    Mat E;
    Mat extrinsics_mat_1;
    Mat extrinsics_mat_2;
};
