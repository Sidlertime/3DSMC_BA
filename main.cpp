#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <map>

#define IMG_WIDTH 640
#define IMG_HEIGHT 480
#define FOCAL 582.871

using namespace cv;
using namespace std;

map<pair<int, int>, int> point_index_map; // Zuordnung von Bild-Feature zu 3D-Punkt

// Calculate Keypoints and Matches
void extractAndMatchFeatures(const Mat& img1, const Mat& img2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& good_matches) {
    Ptr<SIFT> sift = SIFT::create();
    Mat descriptors1, descriptors2;
    
    sift->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    
    if (descriptors1.empty() || descriptors2.empty()) return;
    
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    if (matches.empty()) return;
    
    double min_dist = 100;
    for (auto& m : matches) {
        if (m.distance < min_dist) min_dist = m.distance;
    }
    
    for (auto& m : matches) {
        if (m.distance < 3 * min_dist) {
            good_matches.push_back(m);
        }
    }
}

// Calculate 3D-Points and Camera Pose
void recover3DPointsAndCameraPose(int img1_idx, int img2_idx, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& matches, Mat& global_R, Mat& global_t, vector<Point3f>& points3D, vector<vector<Point2f>>& observations, vector<vector<int>>& point_indices, vector<Mat>& rotations, vector<Mat>& translations) {
    if (matches.empty()) return;
    
    vector<Point2f> pts1, pts2;
    vector<int> matched_indices;
    
    for (auto& match : matches) {
        pts1.push_back(keypoints1[match.queryIdx].pt);
        pts2.push_back(keypoints2[match.trainIdx].pt);
        matched_indices.push_back(match.trainIdx);
    }

    if (pts1.size() < 5) return;
    
    Mat E = findEssentialMat(pts1, pts2, 500.0, Point2d(320, 240), RANSAC);
    Mat R, t;
    recoverPose(E, pts1, pts2, R, t);
    
    // Kumulierte Rotation und Translation berechnen
    global_R = rotations[img1_idx] * R;
    global_t = rotations[img1_idx] * t + translations[img1_idx];
    rotations[img2_idx] = global_R;
    translations[img2_idx] = global_t;
    
    Mat P1 = Mat::eye(3, 4, CV_64F);
    Mat P2(3, 4, CV_64F);
    global_R.copyTo(P2(Range(0, 3), Range(0, 3)));
    global_t.copyTo(P2.col(3));
    
    Mat points4D;
    triangulatePoints(P1, P2, pts1, pts2, points4D);

    if (observations.size() <= img2_idx) {
        observations.resize(img2_idx + 1);
        point_indices.resize(img2_idx + 1);
    }
    
    for (int i = 0; i < points4D.cols; i++) {
        Point3f pt;
        pt.x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
        pt.y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
        pt.z = points4D.at<float>(2, i) / points4D.at<float>(3, i);
        
        pair<int, int> key = {img1_idx, matched_indices[i]};
        if (point_index_map.find(key) == point_index_map.end()) {
            point_index_map[key] = points3D.size();
            points3D.push_back(pt);
        }
        
        observations[img2_idx].push_back(pts2[i]);
        point_indices[img2_idx].push_back(point_index_map[key]);
    }
}

// Save Data to BAL-Format
void saveToBAL(const vector<Mat>& rotations, const vector<Mat>& translations, const vector<Point3f>& points3D, const vector<vector<Point2f>>& observations, const vector<vector<int>>& point_indices, const string& filename) {
    ofstream file(filename);
    if (!file) {
        cerr << "Fehler beim Öffnen der Datei!" << endl;
        return;
    }
    
    int num_cameras = rotations.size();
    int num_points = points3D.size();
    int num_observations = 0;
    for (const auto& obs : observations) {
        num_observations += obs.size();
    }
    
    file << num_cameras << " " << num_points << " " << num_observations << endl;
    
    for (size_t i = 0; i < observations.size(); i++) {
        for (size_t j = 0; j < observations[i].size(); j++) {
            file << i << " " << point_indices[i][j] << " " << observations[i][j].x << " " << observations[i][j].y << endl;
        }
    }
    
    for (size_t i = 0; i < rotations.size(); i++) {
        Mat R, rvec;
        Rodrigues(rotations[i], rvec);
        for (int j = 0; j < 3; j++) file << rvec.at<double>(j, 0) << endl;
        for (int j = 0; j < 3; j++) file << translations[i].at<double>(j, 0) << endl;
        file << FOCAL << endl;
        file << IMG_WIDTH / 2 << endl;
        file << IMG_HEIGHT / 2 << endl;
    }
    
    for (const auto& p : points3D) {
        file << p.x << endl << p.y << endl << p.z << endl;
    }
    
    file.close();
}

int main(int argc, char** argv) {
    string setName = "../Data/Bundle Fusion/apt0";
    int num_images = 10;
    if(argc > 1) setName = argv[1];
    if(argc > 2) num_images = atoi(argv[2]);

    bool compare_all = false; // true = compare every image with every other image, false = just consecutive pairs
    vector<Mat> rotations = { Mat::eye(3, 3, CV_64F) };
    vector<Mat> translations = { Mat::zeros(3, 1, CV_64F) };
    vector<Point3f> points3D;
    vector<vector<Point2f>> observations;
    vector<vector<int>> point_indices;
    
    for (int i = 0; i < num_images - 1; i++) {
        for (int j = i + 1; j < num_images; j++) {
            if (!compare_all && j != i + 1) continue; // Falls nicht alle verglichen werden sollen
            
            string img1_path = setName + "/frame-" + to_string(i).insert(0, 6 - to_string(i).length(), '0') + ".color.jpg";
            string img2_path = setName + "/frame-" + to_string(j).insert(0, 6 - to_string(j).length(), '0') + ".color.jpg";
            
            Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
            Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);
            if (img1.empty() || img2.empty()) {
                cerr << "Fehler beim Laden der Bilder!" << endl;
                return -1;
            }
            
            vector<KeyPoint> keypoints1, keypoints2;
            vector<DMatch> matches;
            extractAndMatchFeatures(img1, img2, keypoints1, keypoints2, matches);

            if (matches.empty()) {
                cerr << "Keine gültigen Matches zwischen " << i << " und " << j << endl;
                continue;
            }
            
            recover3DPointsAndCameraPose(i, j, keypoints1, keypoints2, matches, rotations[j], translations[j], points3D, observations, point_indices, rotations, translations);
        }
    }
    
    saveToBAL(rotations, translations, points3D, observations, point_indices, "output.bal");
    cout << "BAL-Datei gespeichert: output.bal" << endl;
    return 0;
}
