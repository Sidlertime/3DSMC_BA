#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using std::vector;
using std::pair;
using cv::KeyPoint;

class PointAppearance{
public:
    PointAppearance(): point_keys(0, pair<int, KeyPoint>()){};
    PointAppearance(const PointAppearance& pa): point_keys(pa.point_keys.size(), pair<int, KeyPoint>()){
        for (int i = 0; i < point_keys.size(); ++i){
            point_keys[i] = pa.point_keys[i];
        }
    }
    ~PointAppearance(){
        point_keys.clear();
    }

    void add_keypoint(const pair<int, KeyPoint>& kp){
        point_keys.push_back(kp);
    }
    void add_keypoint(const int img_idx, const KeyPoint& kp){
        point_keys.push_back(pair<int, KeyPoint>(img_idx, kp));
    }

    void rm_keypoint(int idx){
        point_keys.erase(point_keys.begin() + idx);
    }

    const pair<int, KeyPoint>&operator[](int idx) const{
        return point_keys[idx];
    };

    const int size() const{
        return point_keys.size();
    }

    static bool has_overlaps(const PointAppearance& pa_1, const PointAppearance& pa_2){
        for (int i = 0; i < pa_1.size(); ++i){
            for (int j = 0; j < pa_2.size(); ++j){
                auto& k1 = pa_1[i];
                auto& k2 = pa_2[i];
                if(k1.first == k2.first && k1.second.pt == k2.second.pt){
                    return true;
                }
            }
        }
        return false;
    }

    static void find_overlaps(const PointAppearance& pa_1, const PointAppearance& pa_2, vector<pair<int, int>>& idx_pairs_out){
        idx_pairs_out.clear();
        for (int i = 0; i < pa_1.size(); ++i){
            for (int j = 0; j < pa_2.size(); ++j){
                auto& k1 = pa_1[i];
                auto& k2 = pa_2[j];
                if(k1.first == k2.first && k1.second.pt == k2.second.pt){
                    idx_pairs_out.push_back(pair<int, int>(i, j));
                }
            }
        }
    }

    vector<pair<int, KeyPoint>> point_keys;     // List of a points Keypoints in the corresponding Images
};
