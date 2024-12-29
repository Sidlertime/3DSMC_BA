#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "ceres/ceres.h"
#include <ceres/rotation.h>
#include <math.h>

using namespace std;
using namespace cv;



struct ReprojectionError {
    ReprojectionError(const Point2f& observed_point, const Point3f& world_point)
        : observed_point_(observed_point), world_point_(world_point) {}

    template <typename T>
    bool operator()(const T* const camera_params, T* residuals) const {
        T predicted[2];
        T point[3] = {T(world_point_.x), T(world_point_.y), T(world_point_.z)};

        // Apply rotation (camera_params[0:2]) and translation (camera_params[3:5])
        T p[3];
        ceres::AngleAxisRotatePoint(camera_params, point, p);
        p[0] += camera_params[3];
        p[1] += camera_params[4];
        p[2] += camera_params[5];

        // Project to image plane
        predicted[0] = p[0] / p[2];
        predicted[1] = p[1] / p[2];

        // Compute residuals
        residuals[0] = predicted[0] - T(observed_point_.x);
        residuals[1] = predicted[1] - T(observed_point_.y);

        return true;
    }

private:
    const Point2f observed_point_;
    const Point3f world_point_;
};


int main ( int argc, char** argv )
{


    //put images into 1 directory level above build folder
    //TODO: Automate reading of all images from directory
    Mat img_1 = imread ("../frame-000000.color.jpg", IMREAD_COLOR );
    Mat img_2 = imread ("../frame-000001.color.jpg", IMREAD_COLOR );




    Mat gray1, gray2;
    cvtColor(img_1, gray1, COLOR_BGR2GRAY);
    cvtColor(img_2, gray2, COLOR_BGR2GRAY);

    // SIFT feature detection
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    sift->detectAndCompute(gray1, noArray(), keypoints_1, descriptors_1);
    sift->detectAndCompute(gray2, noArray(), keypoints_2, descriptors_2);

    // Ensure descriptors are CV_32F
    if (descriptors_1.type() != CV_32F) {
        descriptors_1.convertTo(descriptors_1, CV_32F);
    }
    if (descriptors_2.type() != CV_32F) {
        descriptors_2.convertTo(descriptors_2, CV_32F);
    }


    Mat outimg1;
    drawKeypoints( img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("SIFT_Keypoints",outimg1);

    // Feature matching
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);




   //-- Step 4: Filter matching point pairs
    double min_dist=10000, max_dist=0;

    //Find the minimum distance and maximum distance between all matches, that is, the distance between the most similar and the least similar two sets of points
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //When the distance between descriptors is greater than twice the minimum distance, the matching is considered incorrect. But sometimes the minimum distance will be very small, and an empirical value of 30 is set as the lower limit.
    vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    //-- Step 5: Draw matching results
    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch );
    imshow ( "Match", img_match );
    imshow ( "Good_Match", img_goodmatch );
    waitKey(0);


    sort(good_matches.begin(), good_matches.end(), [](const DMatch &a, const DMatch &b) {
    return a.distance < b.distance;
    });

    vector<Point2f> points1, points2;

    for (size_t i = 0; i < 8; ++i) { //matches.size()

        const auto& match = good_matches[i];

        // Get the keypoints from the match
        const KeyPoint& kp1 = keypoints_1[match.queryIdx];
        const KeyPoint& kp2 = keypoints_2[match.trainIdx];

        points1.push_back(kp1.pt);
        points2.push_back(kp2.pt);

        // Print the coordinates of the matched keypoints
        cout << "Match " << i << ":\n";
        cout << "  Keypoint from Image 1: (" << kp1.pt.x << ", " << kp1.pt.y << ")\n";
        cout << "  Keypoint from Image 2: (" << kp2.pt.x << ", " << kp2.pt.y << ")\n";
    }



    //hard coded from info.txt file in BundleFusion Dataaset
    //TODO: automate reading of intrinsics from the info.txt file
    Mat K = (Mat_<double>(3, 3) << 582.871, 0, 320, 0, 0, 582.871, 240, 0, 0, 0, 1, 0, 0, 0, 0, 1);


    //-- Step 6: // Compute the Fundamental and Essential Matrix using the 8-point algorithm
    Mat F = findFundamentalMat(points1, points2, FM_8POINT);

    cout << "Fundamental Matrix:\n" << F << endl;

    // Compute the Essential Matrix: E = K'^T * F * K
    Mat E = K.t() * F * K;

    cout << "Essential Matrix:\n" << E << endl;

    // // Decompose the Essential Matrix into R and t
    // Mat R1, R2, t;
    // decomposeEssentialMat(E, R1, R2, t);

    // // Output the results
    // cout << "Rotation Matrix R1:\n" << R1 << endl;
    // cout << "Rotation Matrix R2:\n" << R2 << endl;
    // cout << "Translation Vector t:\n" << t << endl;

    //-- Step 7: Recover pose (R, t)
    Mat R, t;
    recoverPose(E, points1, points2, K, R, t);

    cout << "Rotation Matrix R:\n" << R << endl;
    cout << "Translation Vector t:\n" << t << endl;



    //-- Step 8: Generate 3D points via triangulation
    Mat points4D;
    triangulatePoints(K * Mat::eye(3, 4, CV_64F), K * (Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
                                                          R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
                                                          R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2)),
                        points1, points2, points4D);




    // Convert points to homogeneous coordinates
    vector<Point3f> points3D;
    for (int i = 0; i < points4D.cols; i++) {
        Point3f pt(points4D.at<float>(0, i) / points4D.at<float>(3, i),
                       points4D.at<float>(1, i) / points4D.at<float>(3, i),
                       points4D.at<float>(2, i) / points4D.at<float>(3, i));
        points3D.push_back(pt);
    }


    // Initialize camera parameters for Ceres (rotation in angle-axis + translation)
    double camera_params[6] = {0.0, 0.0, 0.0, t.at<double>(0), t.at<double>(1), t.at<double>(2)};


    //-- Step 9:  Setup Ceres problem

    /*TODO: We need 2 ceres optimizers - one for camera pose, one for 3d points. Then we have to setup a loop to iterative over the optimizers alternately. 
            The current single optimizer for camera pose fails in the residual calculation. This needs to be debugged 
    */
    ceres::Problem problem;

    for (size_t i = 0; i < points1.size(); i++) {
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
            new ReprojectionError(points1[i], points3D[i]));
        problem.AddResidualBlock(cost_function, nullptr, camera_params);
    }

    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Output results
    cout << "Final Camera Parameters: ";
    for (double param : camera_params) {
        cout << param << " ";
    }
    cout << endl;

    cout << summary.FullReport() << endl;

    return 0;
}
