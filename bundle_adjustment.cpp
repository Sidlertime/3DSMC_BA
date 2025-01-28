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
#include "bf_preprocessor.hpp"
#include "dataloader_bal.hpp"
#include "mesh_utils.hpp"
#include "solver_ceres.hpp"

using namespace std;
using namespace cv;

#define DEBUG 0


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
        // cout<<"---------------inside ceres rotated points are "<<p[0]<<" "<<p[1]<<" "<<p[2]<<"\n";
        p[0] -= translation[0];
        p[1] -= translation[1];
        p[2] -= translation[2];

        T result[3]; 
        for (int i = 0; i < 3; ++i) {
            result[i] = T(0);
            for (int j = 0; j < 3; ++j) {
                result[i] += K[i * 3 + j] * point[j];
            }
        }
        // cout<<"---------------inside ceres result[0] size is "<<sizeof(result[0])<<"\n";
        // cout<<"---------------inside ceres result points are "<<result[0]<<" "<<result[1]<<" "<<result[2]<<"\n";
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

// Get ground truth position for a given image point
Point3f groundTruthPoint(Point2f& point, Mat& depthMap, int depthShift, Mat& depthIntrinsicInv, Mat& trajectoryInv){
    double z = double(depthMap.at<uint16_t>(int(point.y), int(point.x))) / double(depthShift);
    if (z == 0.0) {
        return Point3f(0.f,0.f,0.f);
    }
    Mat cameraCoord = z * depthIntrinsicInv * Vec3d(double(point.x), double(point.y), 1.0);
    Mat worldCoord = trajectoryInv(Range(0, 3), Range(0, 3)) * (cameraCoord - trajectoryInv(Range(0,3), Range(3,4)));
    return Point3f(worldCoord.at<double>(0), worldCoord.at<double>(1), worldCoord.at<double>(2));
}

vector<Vertex> imageTo3D(Mat& colorMap, Mat& depthMap, int depthShift, Mat&depthIntrinsicsInv, Mat& trajectoryInv){
    vector<Vertex> vertices = vector<Vertex>();

    int width = min(colorMap.cols, depthMap.cols);
    int height = min(colorMap.rows, depthMap.rows);

    cout << "Transforming image (" << width << ", " << height << ") to 3D" << endl;

    for (int x = 0; x < width; x++){
        for (int y = 0; y < height; y++){
            uint16_t& d = depthMap.at<uint16_t>(y, x);
            if(d > 0){
                auto p2d = Point2f(x, y);
                auto point = groundTruthPoint(p2d , depthMap, depthShift, depthIntrinsicsInv, trajectoryInv);
                auto& color = colorMap.at<Vec3b>(y, x);
                vertices.push_back(Vertex(point, color));
            }
        }
    }

    return vertices;
}

int main ( int argc, char** argv )
{
    google::InitGoogleLogging(argv[0]);

    string setName = "apt0";
    int imgSize = 10;
    int every = 1;
    if(argc > 1) setName = argv[1];
    if(argc > 2) imgSize = atoi(argv[2]);
    if(argc > 3) every = atoi(argv[3]);

    if(setName.find("BAL") < setName.length()){
        BA_problem problem = BA_problem();
        load_bal(setName, problem);
        int removed = BARemoveOutliersRelativ(problem, 2.0);
        cout << "Removed " << removed << " Outliers before optimization" << endl;
        balToMesh(problem, "bal.off");

        solveBA(problem);
        removed = BARemoveOutliersRelativ(problem, 1.7);
        cout << "Removed " << removed << " Outliers after optimization" << endl;

        balToMesh(problem, "bal_optimized.off");
        return 0;
    }

    //put images into ../Data/Bundle Fusion/<setName> where setName is i.e. office3
    DataloaderBF loader = DataloaderBF();
    loader.loadImages(setName, imgSize, every); // argv[1] contains the setname, might be generalized to a path in the future
    Mat depthInv = loader.getDepthIntrinsic()(Range(0,3), Range(0,3)).inv();
    cout << "Color intr\n" << loader.getColorIntrinsic() << endl;
    cout << "Depth intr\n" << loader.getDepthIntrinsic() << endl;
    cout << "Depth inverted:\n" << depthInv << endl;

    if(setName == "apt0"){
        BA_problem problem  = processBFSet(loader);

        cout << "Successfully processed Bundle Fusion set " << setName << " with " << problem.num_observations << " Observations" << endl;
        save_bal(problem, "apt0.txt");

        BARemoveOutliersRelativ(problem);

        cout << "Solving the Bundle Adjustment Problem now..." << endl;
        solveBA(problem);
        cout << "Successfully solved the Bundle Adjustment Problem" << endl;

        BARemoveOutliersRelativ(problem);
        balToMesh(problem, "apt0.off");
        return 0;
    }
    
    if(setName == "apt0"){
        auto v = imageTo3D(loader.imagesColor[0],
                    loader.imagesDepth[0],
                    loader.info.depthShift,
                    depthInv,
                    loader.cameraPose[0]);
        cout << v.size() << endl;
        vertexToMesh(v, "frame_0.off");
        return 0;
    }

    if (DEBUG){
        imshow("Colorimage", loader.imagesColor[0]);
        imshow("Depthimage", loader.imagesDepth[0]);

        for (int i = 0; i < loader.nImages; i++){
            cout << "Camera Pose of frame " << i << ":\n" << loader.cameraPose[i] << endl << endl;
        }
    }

    vector<Mat> grays;
    for(int i = 0; i < loader.nImages; i++){
        Mat gray = Mat();
        cvtColor(loader.imagesColor[i], gray, COLOR_BGR2GRAY);
        grays.push_back(gray);
    }

    // SIFT feature detection
    Ptr<SIFT> sift = SIFT::create();

    vector<vector<KeyPoint>> keypoints;
    vector<Mat> descriptors;
    for(int i = 0; i < loader.nImages; i++){
        vector<KeyPoint> kp = vector<KeyPoint>();
        Mat dc = Mat();
        sift->detectAndCompute(grays[i], noArray(), kp, dc);
        sift->clear();

        // Ensure descriptors are CV_32F
        if(dc.type() != CV_32F){
            dc.convertTo(dc, CV_32F);
        }

        keypoints.push_back(kp);
        descriptors.push_back(dc);
    }

    cout << "Finished calculating all SIFT keypoints and descriptors" << endl;


    if(DEBUG){
        Mat outimg1;
        drawKeypoints( loader.imagesColor[0], keypoints[0], outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        imshow("SIFT_Keypoints",outimg1);
    }

    Mat globalRotation = Mat::eye(4, 4, DataType<double>::type);
    vector<Point3f> allPoints;
    vector<Point3f> cameras;
    cameras.push_back(Point3f(0, 0, 0));

    vector<Vertex> all_points_ground_truth;
    all_points_ground_truth.push_back(Vertex(Point3f(0,0,0),Vec3b(255,0,0)));

    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    for (int n = 0; n < loader.nImages - 1; n++){
        // Feature matching
        matches.clear();
        matcher.match(descriptors[n], descriptors[n + 1], matches);


        // -- Step 4: Filter matching point pairs
        double min_dist=10000, max_dist=0;

        // Find the minimum distance and maximum distance between all matches, that is, the distance between the most similar and the least similar two sets of points
        for ( int i = 0; i < descriptors[n].rows; i++ )
        {
            double dist = matches[i].distance;
            if ( dist < min_dist ) min_dist = dist;
            if ( dist > max_dist ) max_dist = dist;
        }

        if(DEBUG) printf("-- Max dist : %f \n", max_dist);
        if(DEBUG) printf("-- Min dist : %f \n", min_dist);

        // When the distance between descriptors is greater than twice the minimum distance, the matching is considered incorrect. But sometimes the minimum distance will be very small, and an empirical value of 30 is set as the lower limit.
        vector< DMatch > good_matches;
        for (int i = 0; i < descriptors[n + 1].rows; i++)
        {
            if (matches[i].distance <= max(2 * min_dist, 30.0))
            {
                good_matches.push_back(matches[i]);
            }
        }

        cout << "Number of good matches are " << good_matches.size() << endl;
        if (good_matches.size() < 8){
            cout << "Not enough Points for matching" << endl;
            continue;
        }

        // -- Step 5: Draw matching results
        if(DEBUG){
            Mat img_match;
            Mat img_goodmatch;
            drawMatches(loader.imagesColor[n], keypoints[n], loader.imagesColor[n + 1], keypoints[n + 1], matches, img_match);
            drawMatches(loader.imagesColor[n], keypoints[n], loader.imagesColor[n + 1], keypoints[n + 1], good_matches, img_goodmatch);
            imshow("Match", img_match);
            imshow("Good_Match", img_goodmatch);
            waitKey(0);
        }


        sort(good_matches.begin(), good_matches.end(), [](const DMatch &a, const DMatch &b) {
            return a.distance < b.distance;
        });

        vector<Point2f> points1, points2;

        int num_points_added = 0;
        int index = 0;

        while (num_points_added < 8) { 
            if(index >= good_matches.size()) {
                cout << "Failed to find 8 Points for 8-Point-Algorithm" << endl;
                break;
            }
            // int random_index = rand() % 20;
            const auto& match = good_matches[index];
            index++;

            // Get the keypoints from the match
            if(match.queryIdx >= keypoints[n].size() || match.trainIdx >= keypoints[n + 1].size()) {
                cout << "Match too large: " << match.queryIdx << ", " << match.trainIdx << endl;
                continue;
            }

            const KeyPoint& kp1 = keypoints[n][match.queryIdx];
            const KeyPoint& kp2 = keypoints[n + 1][match.trainIdx];

            //points1.push_back(kp1.pt);
            //points2.push_back(kp2.pt);

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
                if(DEBUG) cout << "Points: " << points1.size() << endl;
                points1.push_back(kp1.pt);
                points2.push_back(kp2.pt);

                // Print the coordinates of the matched keypoints
                std::cout << "Match " << index << ":\n";
                std::cout << "\tKeypoint from Image 1: (" << kp1.pt.x << ", " << kp1.pt.y << ")\n";
                std::cout << "\tKeypoint from Image 2: (" << kp2.pt.x << ", " << kp2.pt.y << ")\n";

                num_points_added++;
            }
    
        }

        if(num_points_added < 8) continue;

        if(DEBUG){
            // Draw points and numbers on image1
            for (size_t i = 0; i < points1.size(); ++i) {
                // Draw point
                circle(loader.imagesColor[n], points1[i], 5, Scalar(0, 0, 255), -1); // Red dot

                // Add number label
                putText(loader.imagesColor[n], to_string(i + 1), points1[i] + Point2f(5, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }

            // Draw points and numbers on image2
            for (size_t i = 0; i < points2.size(); ++i) {
                // Draw point
                circle(loader.imagesColor[n + 1], points2[i], 5, Scalar(255, 0, 0), -1); // Blue dot

                // Add number label
                putText(loader.imagesColor[n + 1], to_string(i + 1), points2[i] + Point2f(5, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }

            // Display the images
            imshow("Image 1 with Points", loader.imagesColor[0]);
            imshow("Image 2 with Points", loader.imagesColor[1]);

            // Wait for a key press and save the images if needed
            waitKey(0);
        }


        // Intrinsics of Info.txt file in BundleFusion Dataaset
        Mat K4 = Mat(4, 4, DataType<double>::type, &loader.info.calibrationColorIntrinsic);
        //Mat K = (Mat_<double>(3, 3) << 582.871, 0, 320, 0, 582.871, 240, 0, 0, 1);
        Mat K = K4(Range(0, 3), Range(0, 3));

        //-- Step 6: // Compute the Fundamental and Essential Matrix using the 8-point algorithm
        Mat F = findFundamentalMat(points1, points2, FM_8POINT);

        if(DEBUG) cout << "Fundamental Matrix:\n" << F << endl;
        if (F.type() != DataType<double>::type){
            cout << "Error while finding Fundamental Matrix" << endl;
            continue;
        }

        // Compute the Essential Matrix: E = K'^T * F * K
        Mat E = K.t() * F * K;

        //cout << "Essential Matrix:\n" << E << endl;

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

        if(DEBUG) cout << "Rotation Matrix R:\n" << R << endl;
        if(DEBUG) cout << "Translation Vector t:\n" << t << endl;

        //points1.clear();
        //points2.clear();

        // Vectors to hold matching points
        vector<Point2f> all_points_1, all_points_2;

        // Iterate through matches and extract corresponding points
        for (const auto& match : good_matches) {
            if (match.queryIdx >= keypoints[n].size() || match.trainIdx >= keypoints[n + 1].size()) continue;

            all_points_1.push_back(keypoints[n][match.queryIdx].pt); // Point from keypoints1
            all_points_2.push_back(keypoints[n + 1][match.trainIdx].pt); // Point from keypoints2
        }

        // Calculate ground truth points
        //for (auto& p : all_points_1) {
        for (int i = 0; i < all_points_1.size(); i++){
            auto& p = all_points_1[i];
            if (p.x < 0 || p.x >= loader.info.depthWidth || p.y < 0 || p.y >= loader.info.depthHeight) continue;
            Point3f point = groundTruthPoint(p, loader.imagesDepth[n], loader.info.depthShift, depthInv, loader.cameraPose[n]);
            if (point.x == 0 && point.y == 0 && point.z == 0) continue;
            Vec3b color = loader.imagesColor[n].at<Vec3b>(p.y, p.x);
            all_points_ground_truth.push_back(Vertex(point, color));
        }

        //-- Step 8: Generate 3D points via triangulation
        Mat points4D;
        Mat K_float, T_float;
        K.convertTo(K_float, CV_32F);
        Mat T = (Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
                                        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
                                        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));
        T.convertTo(T_float, CV_32F);
        triangulatePoints(K_float * Mat::eye(3, 4, DataType<float>::type), K_float * T_float,
                                                            all_points_1, all_points_2, points4D);




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
        // double camera_params[6] = {angle_axis[0], angle_axis[1], angle_axis[2], t.at<double>(0), t.at<double>(1), t.at<double>(2)};
        double* camera_params = new double[6]; // Dynamically allocate memory for 6 elements

        camera_params[0] = angle_axis[0];
        camera_params[1] = angle_axis[1];
        camera_params[2] = angle_axis[2];
        camera_params[3] = t.at<double>(0);
        camera_params[4] = t.at<double>(1);
        camera_params[5] = t.at<double>(2);

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

        for (size_t i = 0; i < points3D.size(); i++) {

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                new ReprojectionError(all_points_1[i].x, all_points_2[i].y, K_array));

            // double point_3d[3] = {points3D[i].x, points3D[i].y, points3D[i].z};
            double *point_3d = new double[3];
            point_3d[0] = points3D[i].x;
            point_3d[1] = points3D[i].y;
            point_3d[2] = points3D[i].z;

            problem.AddResidualBlock(cost_function, nullptr, camera_params, point_3d);
        }

        // Configure solver
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Output results
        cout << "Final Camera Parameters: ";
        for (int i=0; i<6; i++) {
            cout << camera_params[i] << " ";
        }
        cout << endl;

        if(DEBUG) cout << summary.FullReport() << endl;
        else cout << summary.BriefReport() << endl;

        //all_points_1.clear();
        //all_points_2.clear();


        cv::Vec3d final_angle_axis(camera_params[0], camera_params[1], camera_params[2]);
        cv::Mat rotation_matrix;
        cv::Rodrigues( final_angle_axis, rotation_matrix);

        // Display the rotation matrix
        if(DEBUG) std::cout << "Rotation Matrix:\n" << rotation_matrix << std::endl;

        t.at<double>(0) = camera_params[3];
        t.at<double>(1) = camera_params[4];
        t.at<double>(2) = camera_params[5];
        if(DEBUG) cout << "Translation Vector t:\n" << t << endl;

        cout << "Finished comparing Images " << n << " and " << n + 1 << endl;

        // manually correct the camera pose
        auto& cp = loader.cameraPose[n + 1];
        //cp.copyTo(globalRotation);

        // add new points to old ones
        for(auto& p : points3D){
            Vec3d p_v = cv::Vec3d(p.x, p.y, p.z);
            Mat p_m = globalRotation(Range(0, 3), Range(0, 3)) * p_v + globalRotation(Range(0, 3), Range(3, 4));
            allPoints.push_back(Point3f(p_m.at<double>(0,0), p_m.at<double>(1,0), p_m.at<double>(2,0)));
        }

        cameras.push_back(Point3f(globalRotation.at<double>(0, 3), globalRotation.at<double>(1, 3), globalRotation.at<double>(2, 3)));
        all_points_ground_truth.push_back(Vertex(Point3f(cp.at<double>(0, 3), cp.at<double>(1, 3), cp.at<double>(2, 3)), Vec3b(255, 0, 0)));

        // keep track of global rotation and translation
        Mat transformation = Mat::eye(4, 4, DataType<double>::type);
        rotation_matrix.copyTo(transformation(Range(0, 3), Range(0, 3)));
        t.copyTo(transformation(Range(0, 3), Range(3, 4)));
        globalRotation = transformation * globalRotation;
        if(DEBUG) cout << "Global Transformation:\n" << globalRotation << endl;
    }

    // Write Mesh
    pointsToMesh(allPoints, cameras, "mesh_out.off", 1.0);
    vertexToMesh(all_points_ground_truth, "mesh_out_ref.off");

    waitKey();

    return 0;
}
