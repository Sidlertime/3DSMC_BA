#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen>

using cv::Mat;
using cv::Range;
typedef Eigen::Matrix<float, 3, 3> EMat33f;
typedef Eigen::Matrix<float, 3, 4> EMat34f;
typedef Eigen::Vector3f EVec3f;

EMat34f cv2eigen34f(const Mat& cv){
    assert(cv.rows == 3 && cv.cols == 4);

    EMat34f eigen;
    for (int r = 0; r < 3; ++r){
        for (int c = 0; c < 4; ++c){
            eigen(r, c) = cv.at<float>(r, c);
        }
    }
    return eigen;
}

EMat33f cv2eigen33f(const Mat& cv){
    assert(cv.rows == 3 && cv.cols == 3);

    EMat33f eigen;
    for (int r = 0; r < 3; ++r){
        for (int c = 0; c < 3; ++c){
            eigen(r, c) = cv.at<float>(r, c);
        }
    }
    return eigen;
}

EVec3f cv2eigenVec3f(const Mat& cv){
    assert(cv.rows >= 3 && cv.cols == 1);

    EVec3f eigen;
    float d = 1.0F;
    if(cv.rows >= 4){
        d = cv.at<float>(3);
    }
    if(d == 0) return EVec3f(0, 0, 0);

    for (int i = 0; i < 3; ++i){
        eigen(i) = cv.at<float>(i) / d;
    }
    return eigen;
}

Mat multiple_transforms(const Mat& T1, const Mat& T2){
    Mat T(3, 4, CV_32F);
    Mat t(T1(Range(0, 3), Range(0, 3)) * T2.col(3) + T1.col(3));
    Mat R(T1(Range(0, 3), Range(0, 3)) * T2(Range(0, 3), Range(0, 3)));
    R.copyTo(T(Range(0, 3), Range(0, 3)));
    t.copyTo(T(Range(0, 3), Range(3, 4)));
    return T;
}

EMat34f multiple_transforms(const EMat34f& T1, const EMat34f& T2){
    EMat34f T;
    T.block<3, 3>(0, 0) = T1.block<3, 3>(0, 0) * T2.block<3, 3>(0, 0);
    T.block<3, 1>(0, 3) = T1.block<3, 3>(0, 0) * T2.block<3, 1>(0, 3) + T1.block<3, 1>(0, 3);
    return T;
}

Mat inverse_transform(const Mat& T){
    Mat T_out(3, 4, CV_32F);
    Mat R_inv = T(Range(0, 3), Range(0, 3)).t();
    Mat t_inv = -1.0F * R_inv * T.col(3);
    R_inv.copyTo(T_out(Range(0, 3), Range(0, 3)));
    t_inv.copyTo(T_out(Range(0, 3), Range(3, 4)));
    return T_out;
}

EMat34f inverse_transform(const EMat34f& T){
    EMat34f T_inv;
    T_inv.block<3, 3>(0, 0) = T.block<3, 3>(0, 0).transpose();
    T_inv.block<3, 1>(0, 3) = -1 * (T.block<3, 3>(0, 0).transpose() * T.block<3, 1>(0, 3));
    return T_inv;
}
