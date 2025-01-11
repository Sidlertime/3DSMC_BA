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
#include <cmath>

#include "dataloader_bf.hpp"

using namespace std;
using namespace cv;



// struct ReprojectionError {
//     ReprojectionError(const Point2f& observed_point, const Point3f& world_point)
//         : observed_point_(observed_point), world_point_(world_point) {}

//     template <typename T>
//     bool operator()(const T* const camera_params, T* residuals) const {
//         T predicted[2];
//         T point[3] = {T(world_point_.x), T(world_point_.y), T(world_point_.z)};

//         // Apply rotation (camera_params[0:2]) and translation (camera_params[3:5])
//         T p[3];
//         ceres::AngleAxisRotatePoint(camera_params, point, p);
//         p[0] += camera_params[3];
//         p[1] += camera_params[4];
//         p[2] += camera_params[5];

//         // Project to image plane
//         predicted[0] = p[0] / p[2];
//         predicted[1] = p[1] / p[2];

//         // Compute residuals
//         residuals[0] = predicted[0] - T(observed_point_.x);
//         residuals[1] = predicted[1] - T(observed_point_.y);

//         return true;
//     }

// private:
//     const Point2f observed_point_;
//     const Point3f world_point_;
// };



// Reprojection error cost function
struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y, double* K)
        : observed_x(observed_x), observed_y(observed_y), K(K) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        // Extract rotation (angle-axis) and translation from the camera parameters
        const T* rotation = camera;     // 3 params (angle-axis)
        const T* translation = camera + 3; // 3 params (translation)

        // Transform the 3D point into the camera coordinate system
        T p[3];
        ceres::AngleAxisRotatePoint(rotation, point, p);
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        T result[3]; 
        for (int i = 0; i < 3; ++i) {
            result[i] = T(0);
            for (int j = 0; j < 3; ++j) {
                result[i] += K[i * 3 + j] * point[j];
            }
        }

        // Project the 3D point into the image plane
        T x_image = result[0]/result[2];
        T y_image = result[1]/result[2];


        // Compute residuals
        residuals[0] = x_image - T(observed_x);
        residuals[1] = y_image - T(observed_y);

        return true;
    }

    // static ceres::CostFunction* Create(double observed_x, double observed_y) {
    //     return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
    //         new ReprojectionError(observed_x, observed_y, K));
    // }

private:
    double observed_x, observed_y;
    double* K;
};

double calculateL2Distance(const Point2f& point1, const Point2f& point2) {
    double dx = point2.x - point1.x;
    double dy = point2.y - point1.y;
    return sqrt(dx * dx + dy * dy);
}

int main ( int argc, char** argv )
{
    google::InitGoogleLogging(argv[0]);

    //put images into ../Data/Bundle Fusion/<setName> where setName is i.e. office3
    DataloaderBF loader = DataloaderBF();
    loader.loadImages(argv[1], 10); // argv[1] contains the setname, might be generalized to a path in the future

    for (int i = 0; i < loader.nImages; i++){
        cout << "Camera Pose:\n" << loader.cameraPose[i].at<double>(0,0) << endl << loader.cameraPose[i] << endl << endl;
    }

    vector<Mat> grays;
    for(int i = 0; i < loader.nImages; i++){
        Mat gray;
        cvtColor(loader.imagesColor[i], gray, COLOR_BGR2GRAY);
        grays.push_back(gray);
    }

    // SIFT feature detection
    Ptr<SIFT> sift = SIFT::create();

    vector<vector<KeyPoint>> keypoints;
    vector<Mat> descriptors;
    for(int i = 0; i < loader.nImages; i++){
        vector<KeyPoint> kp;
        Mat dc;
        sift->detectAndCompute(grays[i], noArray(), kp, dc);

        // Ensure descriptors are CV_32F
        if(dc.type() != CV_32F){
            dc.convertTo(dc, CV_32F);
        }

        keypoints.push_back(kp);
        descriptors.push_back(dc);
    }


    Mat outimg1;
    //drawKeypoints( img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawKeypoints( loader.imagesColor[0], keypoints[0], outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("SIFT_Keypoints",outimg1);

    // Feature matching
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    //matcher.match(descriptors_1, descriptors_2, matches);
    matcher.match(descriptors[0], descriptors[1], matches);



   //-- Step 4: Filter matching point pairs
    double min_dist=10000, max_dist=0;

    //Find the minimum distance and maximum distance between all matches, that is, the distance between the most similar and the least similar two sets of points
    for ( int i = 0; i < descriptors[0].rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //When the distance between descriptors is greater than twice the minimum distance, the matching is considered incorrect. But sometimes the minimum distance will be very small, and an empirical value of 30 is set as the lower limit.
    vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors[0].rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    cout<<"Number of good matches are "<<good_matches.size()<<endl;

    //-- Step 5: Draw matching results
    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( loader.imagesColor[0], keypoints[0], loader.imagesColor[1], keypoints[1], matches, img_match );
    drawMatches ( loader.imagesColor[0], keypoints[0], loader.imagesColor[1], keypoints[1], good_matches, img_goodmatch );
    imshow ( "Match", img_match );
    imshow ( "Good_Match", img_goodmatch );
    waitKey(0);


    sort(good_matches.begin(), good_matches.end(), [](const DMatch &a, const DMatch &b) {
    return a.distance < b.distance;
    });

    vector<Point2f> points1, points2;

    int num_points_added = 0;
    int index = 0;

    while (num_points_added<8) { 

        // int random_index = rand() % 20;
        const auto& match = good_matches[index];
        index++;

        // Get the keypoints from the match
        const KeyPoint& kp1 = keypoints[0][match.queryIdx];
        const KeyPoint& kp2 = keypoints[0][match.trainIdx];

        // points1.push_back(kp1.pt);
        // points2.push_back(kp2.pt);

        // // Print the coordinates of the matched keypoints
        // cout << "Match " << i << ":\n";
        // cout << "  Keypoint from Image 1: (" << kp1.pt.x << ", " << kp1.pt.y << ")\n";
        // cout << "  Keypoint from Image 2: (" << kp2.pt.x << ", " << kp2.pt.y << ")\n";
    

        bool isFarEnough = true;
        for (const auto& addedPoint : points1) {
            if (calculateL2Distance(addedPoint, kp1.pt) <= 10.0) {
                isFarEnough = false;
                break;
            }
        }

        // Add keypoints only if far enough
        if (isFarEnough) {
            points1.push_back(kp1.pt);
            points2.push_back(kp2.pt);

            // Print the coordinates of the matched keypoints
            std::cout << "Match " << index << ":\n";
            std::cout << "  Keypoint from Image 1: (" << kp1.pt.x << ", " << kp1.pt.y << ")\n";
            std::cout << "  Keypoint from Image 2: (" << kp2.pt.x << ", " << kp2.pt.y << ")\n";

            num_points_added++;
        }
   
    }

    // Draw points and numbers on image1
    for (size_t i = 0; i < points1.size(); ++i) {
        // Draw point
        circle(loader.imagesColor[0], points1[i], 5, Scalar(0, 0, 255), -1); // Red dot

        // Add number label
        putText(loader.imagesColor[0], to_string(i + 1), points1[i] + Point2f(5, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }

    // Draw points and numbers on image2
    for (size_t i = 0; i < points2.size(); ++i) {
        // Draw point
        circle(loader.imagesColor[1], points2[i], 5, Scalar(255, 0, 0), -1); // Blue dot

        // Add number label
        putText(loader.imagesColor[1], to_string(i + 1), points2[i] + Point2f(5, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }

    // Display the images
    imshow("Image 1 with Points", loader.imagesColor[0]);
    imshow("Image 2 with Points", loader.imagesColor[1]);

    // Wait for a key press and save the images if needed
    waitKey(0);



    // Intrinsics of Info.txt file in BundleFusion Dataaset
    Mat K4 = Mat(4, 4, DataType<double>::type, &loader.info.calibrationColorIntrinsic);
    cout << "Done with K4, first element: " << K4.at<double>(0,0) << endl;
    //Mat K = (Mat_<double>(3, 3) << 582.871, 0, 320, 0, 582.871, 240, 0, 0, 1);
    Mat K = K4(Range(0, 3), Range(0, 3));
    cout << "Done with K (size " << K.size() << "), first element: " << K.at<double>(0,0) << endl;

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
                        //image_1_goodmatches, image_2_goodmatches, points4D); 



    // Convert points to homogeneous coordinates
    vector<Point3f> points3D;
    for (int i = 0; i < points4D.cols; i++) {
        Point3f pt(points4D.at<float>(0, i) / points4D.at<float>(3, i),
                       points4D.at<float>(1, i) / points4D.at<float>(3, i),
                       points4D.at<float>(2, i) / points4D.at<float>(3, i));
        points3D.push_back(pt);
    }


    // Create a double array to store the matrix data
    double rotation_params[9] = {R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2)};

    double angle_axis[3];

    // Convert the rotation matrix to angle-axis
    ceres::RotationMatrixToAngleAxis(rotation_params, angle_axis);


    // Initialize camera parameters for Ceres (rotation in angle-axis + translation)
    double camera_params[6] = {angle_axis[0], angle_axis[1], angle_axis[2], t.at<double>(0), t.at<double>(1), t.at<double>(2)};

    double K_array[9];
    for (int i = 0; i < K.rows; ++i) {
        for (int j = 0; j < K.cols; ++j) {
            K_array[i * K.cols + j] = K.at<double>(i, j);
        }
    }

    //-- Step 9:  Setup Ceres problem

    /*TODO: The current single optimizer for camera pose fails in the residual calculation. This needs to be debugged 
    */
    ceres::Problem problem;

    // for (size_t i = 0; i < points1.size(); i++) {

    //     ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, ceres::DYNAMIC, 6, 3>(
    //         new ReprojectionError(points1[i].x, points1[i].y, K_array));

    //     double point_3d[3] = {points3D[i].x, points3D[i].y, points3D[i].z};
    //     problem.AddResidualBlock(cost_function, nullptr, camera_params, point_3d);
    // }

    for (size_t i = 0; i < points1.size(); i++) {
        double point_3d[3] = {points3D[i].x, points3D[i].y, points3D[i].z};
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                new ReprojectionError(points1[i].x, points1[i].y, K_array)
            ), nullptr, camera_params, point_3d);
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


    cv::Vec3d final_angle_axis(camera_params[0], camera_params[1], camera_params[2]);
    cv::Mat rotation_matrix;
    cv::Rodrigues( final_angle_axis, rotation_matrix);

    // Display the rotation matrix
    std::cout << "Rotation Matrix:\n" << rotation_matrix << std::endl;

    return 0;
}
