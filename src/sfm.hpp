#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "ceres/ceres.h"
#include "Eigen"

#include "util/comparison_pair.hpp"
#include "util/comparison_graph.hpp"
#include "../ba_types.hpp"
#include "util/logging.hpp"
#include <ceres/rotation.h>

using std::vector;
using cv::Mat;
using cv::KeyPoint;

struct SFM_params {
    vector<Mat>& images;
    float focal;
    int width;
    int height;
    bool full_comparison;
    int comparison_window;
    bool use_ref_extrinsics;
    vector<Mat>& ref_extrinsics;
    int match_limit;
};

// Calculate Keypoints and Descriptors
void extract_features(vector<Mat>& imgs, vector<vector<KeyPoint>>& keypoints_out, vector<Mat>& descriptors_out){
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    for (int i = 0; i < imgs.size(); ++i){
        vector<KeyPoint> kp = vector<KeyPoint>();
        Mat dc = Mat();
        sift->detectAndCompute(imgs[i], cv::noArray(), kp, dc);
        sift->clear();

        // Ensure descriptors are CV_32F
        if(dc.type() != CV_32F){
            dc.convertTo(dc, CV_32F);
        }

        keypoints_out[i] = kp;
        descriptors_out[i] = dc;
    }
}

// Compute Matches
void match_features(vector<Mat>& descriptors, vector<vector<KeyPoint>>& keypoints, vector<ComparisonPair>& pairs_out, SFM_params& params){
    const int nImages = descriptors.size();
    cv::BFMatcher matcher(cv::NORM_L2);

    for(int i = 0; i < nImages - 1; ++i){
        for (int j = i + 1; j < nImages; ++j){
            if(!params.full_comparison && j > i + params.comparison_window) break;   // only compare two following images

            vector<DMatch> matches;
            matcher.match(descriptors[i], descriptors[j], matches, cv::noArray());

            // Filter matches
            double min_dist=10000, max_dist=0;

            // Find the minimum distance and maximum distance between all matches, that is, the distance between the most similar and the least similar two sets of points
            for ( int i = 0; i < matches.size(); i++ )
            {
                double dist = matches[i].distance;
                if ( dist < min_dist ) min_dist = dist;
                if ( dist > max_dist ) max_dist = dist;
            }

            // When the distance between descriptors is greater than twice the minimum distance, the matching is considered incorrect. But sometimes the minimum distance will be very small, and an empirical value of 30 is set as the lower limit.
            vector< DMatch > good_matches;
            for (int i = 0; i < matches.size(); i++)
            {
                if (matches[i].distance <= std::max(2 * min_dist, 50.0))
                {
                    good_matches.push_back(matches[i]);
                }
            }

            if(good_matches.size() > params.match_limit) good_matches.resize(params.match_limit);
            else if(matches.size() > params.match_limit) matches.resize(params.match_limit);

            if (good_matches.size() >= 8) pairs_out.push_back(ComparisonPair(i, j, good_matches, keypoints[i], keypoints[j]));
            else pairs_out.push_back(ComparisonPair(i, j, matches, keypoints[i], keypoints[j]));
        }
    }
}

// Estimate Pose
void estimate_pose(ComparisonPair& pair, bool use_ref_pose = false, Mat ref_pose1 = Mat(), Mat ref_pose2 = Mat()){
    cout << "Pair " << pair.cam_idx_1 << ", " << pair.cam_idx_2 << endl;
    if (use_ref_pose && ref_pose1.cols >= 4 && ref_pose1.rows >= 3 && ref_pose2.cols >= 4 && ref_pose2.rows >= 3){
        Mat R1_inv, R2_inv;
        ref_pose1(Range(0, 3), Range(0, 4)).copyTo(R1_inv);
        ref_pose2(Range(0, 3), Range(0, 4)).copyTo(R2_inv);
        if (R1_inv.type() != CV_32F){
            R1_inv.convertTo(R1_inv, CV_32F);
        }
        if (R2_inv.type() != CV_32F){
            R2_inv.convertTo(R2_inv, CV_32F);
        }
        Mat R21 = multiple_transforms(R1_inv, inverse_transform(R2_inv));
        R21.copyTo(pair.extrinsics_mat_2);
        return;
    }

    if(pair.matches.size() < 8) {
        cout << "Not enough Points in Pair " << pair.cam_idx_1 << ", " << pair.cam_idx_2 << endl;
        return;     // Not enough points for Fundamental computation with 8-point algorithm
    }

    vector<cv::Point2f> points_1(pair.matches.size()), points_2(pair.matches.size());
    for(int i = 0; i < pair.matches.size(); ++i){
        points_1[i] = pair.keypoints_1[pair.matches[i].queryIdx].pt;
        points_2[i] = pair.keypoints_2[pair.matches[i].trainIdx].pt;
    }

    // Find Fundamental Matrix
    //pair.F = cv::findFundamentalMat(points_1, points_2, cv::FM_8POINT);
    pair.F = cv::findFundamentalMat(points_1, points_2, cv::RANSAC);
    if(pair.F.type() != CV_32F){
        pair.F.convertTo(pair.F, CV_32F);
    }

    // E = K.T * F * K
    pair.E = pair.intrinsics_mat_2.t() * pair.F * pair.intrinsics_mat_1;

    // Recover Pose of Cam-2 relative to Cam-1, assumes both cameras have the same intrinsics
    Mat R, t, K, E;
    pair.intrinsics_mat_1.convertTo(K, CV_64F);
    pair.E.convertTo(E, CV_64F);
    recoverPose(E, points_1, points_2, K, R, t);
    for(int r = 0; r < 3; ++r){
        for(int c = 0; c < 3; ++c){
            pair.extrinsics_mat_2.at<float>(r, c) = (float) R.at<double>(r, c);
        }
        pair.extrinsics_mat_2.at<float>(r, 3) = (float) t.at<double>(r);
    }
}

BA_problem sfm_pipeline(SFM_params& params){
    const int nImages = params.images.size();
    cout << "Calculating Keypoints for " << nImages << " images..." << endl;

    // Feature Detection
    vector<vector<KeyPoint>> keypoints(nImages, vector<KeyPoint>());
    vector<Mat> descriptors(nImages);
    extract_features(params.images, keypoints, descriptors);

    if(params.full_comparison) cout << "Matching Features of every image to image..." << endl;
    else cout << "Matching Features of following images..." << endl;
    // Feature Matching
    vector<ComparisonPair> pairs;
    match_features(descriptors, keypoints, pairs, params);
    // Remove Pairs with too few matches
    for (int i = pairs.size() - 1; i >= 0; --i){
        if (pairs[i].matches.empty() || pairs[i].matches.size() < 5){   // have at least 5 for RANSAC
            pairs.erase(pairs.begin() + i);
        }
    }
    cout << "Successfully Matched " << pairs.size() << " Pairs" << endl;

    if (DEBUG){
        Mat matches;
        cv::drawMatches(params.images[0], keypoints[0], params.images[1], keypoints[1], pairs[0].matches, matches);
        cv::imshow("Matches", matches);
        cv::waitKey();
        cv::destroyAllWindows();
    }

    cout << "Estimating Poses for each Pair..." << endl;
    // Pose Estimation
    for (int i = 0; i < pairs.size(); ++i){
        auto& pair = pairs[i];
        pair.update_intrinsics(params.focal, params.focal, params.width/2.0F, params.height/2.0F);
        if (params.use_ref_extrinsics){
            estimate_pose(pair, params.use_ref_extrinsics, params.ref_extrinsics[pair.cam_idx_1], params.ref_extrinsics[pair.cam_idx_2]);
        } else {
            estimate_pose(pair);
        }
    }

    cout << "Triangulating all Points..." << endl;
    // Triangulate Pairs and turn into graphs for combination
    vector<ComparisonGraph> graphs(pairs.size(), ComparisonGraph());
    for (int i = 0; i < pairs.size(); ++i){
        cout << "Pair " << i << endl;
        auto& pair = pairs[i];

        // triangulate points of a pair
        vector<cv::Point2f> points_1, points_2;
        for (auto& match : pair.matches){
            points_1.push_back(pair.keypoints_1[match.queryIdx].pt);
            points_2.push_back(pair.keypoints_2[match.trainIdx].pt);
        }
        cout << "Triangulating " << points_1.size() << " Points" << endl;
        Mat points_tri;
        cv::triangulatePoints(pair.intrinsics_mat_1 * pair.extrinsics_mat_1, pair.intrinsics_mat_2 * pair.extrinsics_mat_2, points_1, points_2, points_tri);
        // remove invalid triangulated points
        vector<int> valid_indices;
        vector<DMatch> valid_matches;
        for (int idx = 0; idx < points_tri.cols; ++idx){
            Eigen::Vector4f p_h = cv2eigenVec3f(points_tri.col(idx)).homogeneous();
            Eigen::Vector3f pc1 = cv2eigen34f(pair.extrinsics_mat_1) * p_h;
            Eigen::Vector3f pc2 = cv2eigen34f(pair.extrinsics_mat_2) * p_h;
            if (pc1.z() > 0 && pc2.z() > 0){
                valid_indices.push_back(idx);
                valid_matches.push_back(pair.matches[idx]);
            }
        }
        pair.matches = valid_matches;

        if(DEBUG && i < 1){
            Mat kpImg, kpImg2;
            vector<KeyPoint> kp_refs, kp_reprojected;
            for(auto& match : pair.matches) kp_refs.push_back(pair.keypoints_1[match.queryIdx]);
            for (int idx = 0; idx < kp_refs.size(); ++idx){
                Eigen::Vector3f proj = cv2eigen33f(pair.intrinsics_mat_1) * cv2eigen34f(pair.extrinsics_mat_1) * cv2eigenVec3f(points_tri.col(idx)).homogeneous();
                Point2f p = Point2f(proj.x() / proj.z(), proj.y() / proj.z());
                cout << "Original Point: " << kp_refs[idx].pt << endl;
                cout << "Reprojected Point: " << p << endl;
                kp_reprojected.push_back(KeyPoint(p, kp_refs[i].size));
            }
            cv::drawKeypoints(params.images[pairs[i].cam_idx_1], kp_refs, kpImg, Scalar(255, 0, 0));
            cv::drawKeypoints(kpImg, kp_reprojected, kpImg2, Scalar(0, 0, 255));
            cv::imshow("Keypoints references", kpImg);
            cv::imshow("Keypoints with reprojection of triangulated points", kpImg2);
            cv::waitKey();
            cv::destroyAllWindows();
        }

        ComparisonGraph g(pair);
        for (int idx = 0; idx < valid_indices.size(); ++idx){
            g.points3d[idx] = cv2eigenVec3f(points_tri.col(valid_indices[idx]));
        }
        graphs[i] = g;
    }
    cout << "Triangulated Points of " << graphs.size() << " graphs" << endl;

    cout << "Merging all graphs together..." << endl;
    // Combine all graphs
    ComparisonGraph full_g(graphs[0]);
    for (int i = 1; i < graphs.size(); ++i){
        ComparisonGraph::merge_graphs(full_g, graphs[i]);
    }

    cout << "Preparing Bundle Adjustment Problem..." << endl;
    // Bundle Adjustment
    BA_problem problem;
    problem.num_cameras = full_g.n_cams;
    problem.num_points = full_g.points3d.size();
    problem.num_observations = 0;
    for (int i = 0; i < full_g.appearances.size(); i++){
        problem.num_observations += full_g.appearances[i].size();
    }
    problem.cameras = new Camera[problem.num_cameras];
    problem.points = new Eigen::Vector3d[problem.num_points];
    problem.invalid_points = new bool[problem.num_points];
    problem.colors = new Eigen::Vector3i[problem.num_points];
    problem.observations = new Observation[problem.num_observations];

    // add each camera
    for (int i = 0; i < problem.num_cameras; ++i){
        Camera& c = problem.cameras[full_g.cams_idx[i]];
        c.cx = (double)params.width / 2.0;
        c.cy = (double)params.height / 2.0;
        c.f = (double)params.focal;
        c.t = full_g.extrinsics[i].block<3, 1>(0, 3).cast<double>();
        Eigen::Matrix3d R = full_g.extrinsics[i].block<3, 3>(0, 0).cast<double>();
        ceres::RotationMatrixToAngleAxis(R.data(), c.R.data());
    }

    // add each points initial 3D coordinates
    for (int i = 0, j = 0; i < problem.num_points; ++i){
        Eigen::Vector3d& p = problem.points[i];
        p = full_g.points3d[i].cast<double>();
        bool& i_p = problem.invalid_points[i];
        i_p = false;

        // for each points, add all its appearances as observations and calculate color
        //cout << full_g.appearances[i].size() << " Observations for Point " << i << endl;
        Eigen::Vector3d color_avg(0, 0, 0);
        for (auto& pk : full_g.appearances[i].point_keys){
            Observation& o = problem.observations[j];
            o.camera_index = pk.first;
            o.point_index = i;
            o.x = (double) pk.second.pt.x;
            o.y = (double) pk.second.pt.y;
            j++;

            // average the colors over all appearances
            auto& p_color = params.images[o.camera_index].at<cv::Vec3b>(o.y, o.x);
            color_avg.x() += p_color(0);
            color_avg.y() += p_color(1);
            color_avg.z() += p_color(2);
        }
        const double n_appearances = (double) full_g.appearances[i].size();
        Eigen::Vector3i& color = problem.colors[i];
        color.x() = (int) (color_avg.x() / n_appearances);
        color.y() = (int) (color_avg.y() / n_appearances);
        color.z() = (int) (color_avg.z() / n_appearances);
    }

    return problem;
}
