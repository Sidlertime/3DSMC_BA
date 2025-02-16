#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "comparison_pair.hpp"
#include "point_appearance.hpp"
#include "matrix_utils.hpp"
#include "sorting.hpp"

using std::vector;
using cv::Mat;

class ComparisonGraph{
public:
    ComparisonGraph(){
        n_cams = 0;
        cams_idx = vector<int>();
        intrinsics = vector<EMat33f>();
        extrinsics = vector<EMat34f>();
        appearances = vector<PointAppearance>();
        points3d = vector<EVec3f>();
    };
    ComparisonGraph(const ComparisonPair& pair){
        n_cams = 2;
        cams_idx.push_back(pair.cam_idx_1);
        cams_idx.push_back(pair.cam_idx_2);
        intrinsics.push_back(cv2eigen33f(pair.intrinsics_mat_1));
        intrinsics.push_back(cv2eigen33f(pair.intrinsics_mat_2));
        extrinsics.push_back(cv2eigen34f(pair.extrinsics_mat_1));
        extrinsics.push_back(cv2eigen34f(pair.extrinsics_mat_2));

        appearances.resize(pair.matches.size(), PointAppearance());
        for (int i = 0; i < pair.matches.size(); ++i){
            appearances[i].add_keypoint(pair.cam_idx_1, pair.keypoints_1[pair.matches[i].queryIdx]);
            appearances[i].add_keypoint(pair.cam_idx_2, pair.keypoints_2[pair.matches[i].trainIdx]);
        }

        points3d.resize(pair.matches.size());
    }
    ComparisonGraph(const ComparisonGraph& g){
        n_cams = g.n_cams;
        cams_idx = g.cams_idx;
        intrinsics = g.intrinsics;
        extrinsics = g.extrinsics;
        appearances.resize(g.appearances.size(), PointAppearance());
        points3d.resize(g.appearances.size());
        for (int i = 0; i < g.appearances.size(); ++i){
            appearances[i] = g.appearances[i];
            points3d[i] = g.points3d[i];
        }
    }
    ~ComparisonGraph(){
        cams_idx.clear();
        intrinsics.clear();
        extrinsics.clear();
        appearances.clear();
        points3d.clear();
    };

    int indexof(int img_idx){
        for (int i = 0; i < n_cams; ++i){
            if (cams_idx[i] == img_idx) {
                return i;
            }
        }
        return -1;
    }

    static void merge_appearance(PointAppearance& pa1, PointAppearance& pa2){
        vector<pair<int, int>> overlaps;
        PointAppearance::find_overlaps(pa1, pa2, overlaps);
        //cout << "Out of " << pa1.size() << " Keypoints in A and " << pa2.size() << " Keypoints in B are " << overlaps.size() << " overlapping" << endl;

        vector<int> idx2;
        for (auto& idx_pair : overlaps){
            idx2.push_back(idx_pair.second);
        }
        for (int i = 0; i < pa2.size(); ++i){
            auto iter = std::find(idx2.begin(), idx2.end(), i);
            if(iter == idx2.end()){
                //cout << "Added Keypoint " << pa2[i].second.pt << " (Image " << pa2[i].first << ")" << endl; 
                pa1.add_keypoint(pa2[i]);
                //cout << "New Size: " << pa1.size() << endl;
            }
        }
    }

    static void merge_graphs(ComparisonGraph& g1, ComparisonGraph& g2){
        //cout << "Merging graph A with " << g1.n_cams << " Cameras and graph B with " << g2.n_cams << " Cameras" << endl;
        vector<int>::iterator iter;

        // get the common cameras of both graphs
        vector<int> cameras_common(std::min(g1.n_cams, g2.n_cams));
        iter = std::set_intersection(g1.cams_idx.begin(), g1.cams_idx.end(), g2.cams_idx.begin(), g2.cams_idx.end(), cameras_common.begin());
        cameras_common.resize(iter - cameras_common.begin());

        if(cameras_common.empty()) {
            cout << "Unable to merge graphs" << endl;
            return;  // nothing to merge with
        }

        // get the different cameras in 2 from 1
        vector<int> cameras_diff(std::max(g1.n_cams, g2.n_cams));
        iter = std::set_difference(g2.cams_idx.begin(), g2.cams_idx.end(), g1.cams_idx.begin(), g1.cams_idx.end(), cameras_diff.begin());
        cameras_diff.resize(iter - cameras_diff.begin());

        // merging graphs along first camera
        int idx_g1 = g1.indexof(cameras_common[0]);
        int idx_g2 = g2.indexof(cameras_common[0]);
        EMat34f extrinsics_1 = g1.extrinsics[idx_g1];
        EMat34f extrinsics_2 = g2.extrinsics[idx_g2];
        EMat34f extrinsics_21 = multiple_transforms(inverse_transform(extrinsics_1), extrinsics_2);
        EMat34f extrinsics_21_inv = inverse_transform(extrinsics_21);

        // rebase points of graph 2 into graph 1 coords
        for (int i = 0; i < g2.points3d.size(); ++i){
            EVec3f& p = g2.points3d[i];
            p = extrinsics_21 * p.homogeneous();
        }

        // append different cameras to graph 1
        for (int i = 0; i < cameras_diff.size(); ++i){
            g1.cams_idx.push_back(cameras_diff[i]);
            const int idx = g2.indexof(cameras_diff[i]);
            g1.intrinsics.push_back(g2.intrinsics[idx]);
            g1.extrinsics.push_back(multiple_transforms(g2.extrinsics[idx], extrinsics_21_inv));
        }

        g1.n_cams = g1.cams_idx.size();
        //cout << "Merged graphs with cameras ";
        //for (auto& c :g1.cams_idx) cout << c << " ";
        //cout << endl;
        vector<size_t> indices;
        sort<int>(g1.cams_idx, g1.cams_idx, indices);
        reorder<EMat33f>(g1.intrinsics, indices, g1.intrinsics);
        reorder<EMat34f>(g1.extrinsics, indices, g1.extrinsics);

        const int n_appearances = g1.appearances.size();
        int merged_appearances = 0;
        for (int j = 0; j < g2.appearances.size(); ++j){
            PointAppearance& pa_2 = g2.appearances[j];
            bool is_connected = false;
            for (int i = 0; i < n_appearances; ++i){
                PointAppearance& pa_1 = g1.appearances[i];
                if (PointAppearance::has_overlaps(pa_1, pa_2)){
                    // merge existing appearances
                    merged_appearances++;
                    merge_appearance(pa_1, pa_2);
                    is_connected = true;
                    break;
                }
            }

            // add new appearances
            if (!is_connected){
                g1.appearances.push_back(pa_2);
                g1.points3d.push_back(g2.points3d[j]);  // coords must be converted as this has already been done earlier
            }
        }
        //cout << "Found " << merged_appearances << " Points already existing and " << g2.appearances.size() - merged_appearances << " new ones" << endl;
    }

    int n_cams;
    vector<int> cams_idx;
    vector<EMat33f> intrinsics;
    vector<EMat34f> extrinsics;
    vector<PointAppearance> appearances;
    vector<EVec3f> points3d;
};
