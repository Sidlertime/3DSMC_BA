#pragma once

#include <map>

#include "ba_types.hpp"
#include "dataloader_bf.hpp"

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

// Calculate Observations and initial Camera Position/Rotation and Point Positions for solving the BA
BA_problem processBFSet(DataloaderBF& loader){

    // SIFT
    cout << "Calculating keypoints and descriptors..." << endl;

    vector<Mat> grays;

    Ptr<SIFT> sift = SIFT::create();

    vector<vector<KeyPoint>> keypoints;
    vector<Mat> descriptors;
    for(int i = 0; i < loader.nImages; i++){
        Mat gray = Mat();
        cvtColor(loader.imagesColor[i], gray, COLOR_BGR2GRAY);
        grays.push_back(gray);

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

    cout << "Finished calculating keypoints and descriptors..." << endl;

    cout << "Comparing keypoints Image to Image" << endl;

    vector<Point3d> points3D;
    vector<int> invalids;
    vector<Observation> observations;
    vector<map<int, int>> point2idx_map;
    point2idx_map.resize(loader.nImages);

    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;

    Mat K_float;
    //loader.colorIntrinsic(Range(0, 3), Range(0, 3)).convertTo(K_float, CV_32F);
    loader.colorIntrinsic(Range(0, 3), Range(0, 3)).convertTo(K_float, CV_64F);
    Mat depth_inv = loader.getDepthIntrinsic()(Range(0,3), Range(0,3)).inv();

    // Compare every frame with every frame for keypoints
    for(int m = 0; m < loader.nImages - 1; m++){
        cout << "Comparing Image " << m << " to all others..." << endl;
        int doubles = 0;
        for (int n = m + 1; n < loader.nImages; n++){
            // Feature matching
            matches.clear();
            matcher.match(descriptors[m], descriptors[n], matches);

            // Filter matching point pairs
            double min_dist=10000, max_dist=0;

            // Find the minimum distance and maximum distance between all matches, that is, the distance between the most similar and the least similar two sets of points
            for ( int i = 0; i < descriptors[m].rows; i++ )
            {
                double dist = matches[i].distance;
                if ( dist < min_dist ) min_dist = dist;
                if ( dist > max_dist ) max_dist = dist;
            }

            vector<DMatch> good_matches;
            for (int i = 0; i < descriptors[n].rows; i++)
            {
                if (matches[i].distance <= max(2 * min_dist, 30.0))
                {
                    good_matches.push_back(matches[i]);
                }
            }

            // Here 8 point algorithm could be used to get the initial Rotation and Translation, we will use given camera pose instead

            vector<Point2f> points_1, points_2;

            for (const auto& match : good_matches)
            {
                if(match.queryIdx >= keypoints[m].size() || match.trainIdx >= keypoints[n].size()) continue;

                try{
                    // we already now this point from image m
                    auto& point_idx = point2idx_map[m].at(match.queryIdx);
                    auto& point = keypoints[n][match.trainIdx].pt;

                    point2idx_map[n][match.trainIdx] = point_idx;
                    observations.push_back(Observation(n, point_idx, point.x, point.y));
                    doubles++;
                } catch (std::out_of_range){
                    try {
                        // we already now this point from image n
                        auto& point_idx = point2idx_map[n].at(match.trainIdx);
                        auto& point = keypoints[m][match.queryIdx].pt;

                        point2idx_map[m][match.queryIdx] = point_idx;
                        observations.push_back(Observation(m, point_idx, point.x, point.y));
                        doubles++;
                    }
                    catch (std::out_of_range){
                        // new point
                        int point_idx = points3D.size() + points_1.size();
                        auto& point_m = keypoints[m][match.queryIdx].pt;
                        auto& point_n = keypoints[n][match.trainIdx].pt;

                        point2idx_map[m][match.queryIdx] = point_idx;
                        point2idx_map[n][match.trainIdx] = point_idx;

                        observations.push_back(Observation(m, point_idx, point_m.x, point_m.y));
                        observations.push_back(Observation(n, point_idx, point_n.x, point_n.y));

                        points_1.push_back(keypoints[m][match.queryIdx].pt); // Point from Image 1
                        points_2.push_back(keypoints[n][match.trainIdx].pt);
                    }
                }
            }
            
            // no new points to triangulate
            if(points_1.size() == 0) continue;

            /* Generate 3D points via triangulation 
            Mat points4D;
            Mat T_m_float, T_n_float;
            //loader.cameraPose[m].convertTo(T_m_float, CV_32F);
            //loader.cameraPose[n].convertTo(T_n_float, CV_32F);
            loader.cameraPose[m].convertTo(T_m_float, CV_64F);
            loader.cameraPose[n].convertTo(T_n_float, CV_64F);

            //vector<Point2f> undistorted_1, undistorted_2;
            //undistortPoints(points_1, undistorted_1, K_float, Vec4d(0, 0, 0, 0));
            //undistortPoints(points_2, undistorted_2, K_float, Vec4d(0, 0, 0, 0));

            triangulatePoints(K_float * T_m_float.inv()(Range(0, 3), Range(0, 4)), 
                    K_float * T_n_float.inv()(Range(0, 3), Range(0, 4)), points_1, points_2, points4D);
            //triangulatePoints(T_m_float(Range(0, 3), Range(0, 4)), 
            //        T_n_float(Range(0, 3), Range(0, 4)), undistorted_1, undistorted_2, points4D);
            
            // Convert points to homogeneous coordinates
            for (int i = 0; i < points4D.cols; i++) {
                if(points4D.at<float>(3, i) == 0) continue;

                Point3f pt(points4D.at<float>(0, i) / points4D.at<float>(3, i),
                            points4D.at<float>(1, i) / points4D.at<float>(3, i),
                            points4D.at<float>(2, i) / points4D.at<float>(3, i));
                points3D.push_back(pt);
            }*/

            /* Using "ground truth" as initial value */
            for(int i = 0; i < points_1.size(); i++){
                auto g_t = groundTruthPoint(points_1[i], loader.imagesDepth[m], loader.info.depthShift, depth_inv, loader.cameraPose[m]);
                if( g_t.x == 0 && g_t.y == 0 && g_t.z == 0){
                    // invalid point, so store index to invalidate later
                    invalids.push_back(points3D.size() + i);
                }
                points3D.push_back(g_t);
            }
        }
        
        cout << "Found " << doubles << " Points already seen in previous comparisons" << endl;
    }

    cout << "Finished comparing all Images" << endl;

    BA_problem problem;
    problem.dynamic_K = false;
    problem.num_cameras = loader.nImages;
    problem.num_observations = observations.size();
    problem.num_points = points3D.size();

    problem.cameras = new Camera[problem.num_cameras];
    for (int i = 0; i < problem.num_cameras; i++){
        Mat c_p = loader.cameraPose[i];
        auto& c = problem.cameras[i];
        Mat R;
        cv::Rodrigues(c_p(Range(0, 3), Range(0, 3)), R);
        c.R = Eigen::Vector3d(R.at<double>(0), R.at<double>(1), R.at<double>(2));
        c.t = Eigen::Vector3d(c_p.at<double>(0, 3), c_p.at<double>(1, 3), c_p.at<double>(2, 3));
        c.f = loader.colorIntrinsic.at<double>(0,0);
        c.k1 = loader.colorIntrinsic.at<double>(0,2);
        c.k2 = loader.colorIntrinsic.at<double>(1,2);
        //cout << "Camera Pose:\n" << c_p << endl;
        //cout << "Camera: " << i << " with Rotation: " << c.R << " and Translation: " << c.t << endl;
    }

    problem.observations = new Observation[problem.num_observations];
    for (int i = 0; i < problem.num_observations; i++){
        auto& obs = observations[i];
        if(obs.camera_index >= problem.num_cameras){
            cout << "PANIK with observation " << i << " has camera index " << obs.camera_index << endl;
        }
        if(obs.point_index >= problem.num_points){
            cout << "PANIK with observation " << i << " has point index " << obs.point_index << endl;
        }
        problem.observations[i] = Observation(obs);
    }

    problem.points = new Eigen::Vector3d[problem.num_points];
    for (int i = 0; i < problem.num_points; i++){
        auto& p = points3D[i];
        problem.points[i] = Eigen::Vector3d(p.x, p.y, p.z);
    }

    problem.invalid_points = new bool[problem.num_points];
    for (auto& idx : invalids){
        problem.invalid_points[idx] = true;
    }

    return problem;
}
