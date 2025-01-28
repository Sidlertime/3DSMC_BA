#pragma once

#include <map>

#include "ba_types.hpp"
#include "dataloader_bf.hpp"

BA_problem processBFSet(DataloaderBF& loader){

    // SIFT
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

    vector<Point3d> points3D;
    vector<Observation> observations;
    vector<map<int, int>> point2idx_map;
    point2idx_map.resize(loader.nImages);

    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;

    // Compare every frame with every frame for keypoints
    for(int m = 0; m < loader.nImages - 1; m++){
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

                    point2idx_map[n].at(match.queryIdx) = point_idx;
                    observations.push_back(Observation(n, point_idx, point.x, point.y));
                } catch (std::out_of_range){
                    try {
                        // we already now this point from image n
                        auto& point_idx = point2idx_map[n].at(match.trainIdx);
                        auto& point = keypoints[m][match.queryIdx].pt;

                        point2idx_map[m].at(match.queryIdx) = point_idx;
                        observations.push_back(Observation(m, point_idx, point.x, point.y));
                    }
                    catch (std::out_of_range){
                        // new point
                        int point_idx = points3D.size() + points_1.size();
                        auto& point_m = keypoints[m][match.queryIdx].pt;
                        auto& point_n = keypoints[n][match.trainIdx].pt;

                        observations.push_back(Observation(m, point_idx, point_m.x, point_m.y));
                        observations.push_back(Observation(n, point_idx, point_n.x, point_n.y));

                        points_1.push_back(keypoints[m][match.queryIdx].pt); // Point from Image 1
                        points_2.push_back(keypoints[n][match.trainIdx].pt);
                    }
                }
            }
            

            // Generate 3D points via triangulation
            Mat points4D;
            Mat K_float, T_m_float, T_n_float;
            loader.colorIntrinsic(Range(0, 3), Range(0, 3)).convertTo(K_float, CV_32F);
            loader.cameraPose[m].convertTo(T_m_float, CV_32F);
            loader.cameraPose[n].convertTo(T_n_float, CV_32F);

            triangulatePoints(K_float * T_m_float(Range(0, 3), Range(0, 4)), 
                    K_float * T_n_float(Range(0, 3), Range(0, 4)), points_1, points_2, points4D);
            
            // Convert points to homogeneous coordinates
            for (int i = 0; i < points4D.cols; i++) {
                Point3f pt(points4D.at<float>(0, i) / points4D.at<float>(3, i),
                            points4D.at<float>(1, i) / points4D.at<float>(3, i),
                            points4D.at<float>(2, i) / points4D.at<float>(3, i));
                points3D.push_back(pt);
            }
        }
    }

    BA_problem problem;
    problem.num_cameras = loader.nImages;
    problem.num_observations = observations.size();
    problem.num_points = points3D.size();

    problem.cameras = new Camera[problem.num_cameras];
    for (int i = 0; i < problem.num_cameras; i++){
        Mat c_p = loader.cameraPose[i].inv();
        auto& c = problem.cameras[i];
        Mat R;
        cv::Rodrigues(c_p(Range(0, 3), Range(0, 3)), R);
        c.R = Eigen::Vector3d(R.at<double>(0), R.at<double>(1), R.at<double>(2));
        c.t = Eigen::Vector3d(c_p.at<double>(0, 3), c_p.at<double>(1, 3), c_p.at<double>(2, 3));
        c.f = loader.colorIntrinsic.at<double>(0,0);
        c.k1 = loader.colorIntrinsic.at<double>(0,2);
        c.k2 = loader.colorIntrinsic.at<double>(1,2);
        cout << "Camera Pose:\n" << c_p << endl;
        cout << "Camera: " << i << " with Rotation: " << c.R << " and Translation: " << c.t << endl;
    }

    problem.observations = new Observation[problem.num_observations];
    for (int i = 0; i < problem.num_observations; i++){
        auto& obs = observations[i];
        problem.observations[i] = obs;
    }

    problem.points = new Eigen::Vector3d[problem.num_points];
    for (int i = 0; i < problem.num_points; i++){
        auto& p = points3D[i];
        problem.points[i] = Eigen::Vector3d(p.x, p.y, p.z);
    }

    return problem;
}
