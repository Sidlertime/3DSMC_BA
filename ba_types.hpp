#pragma once

#include "Eigen"

struct Observation {
    int camera_index;
    int point_index;
    double x;
    double y;

    Observation(int c_idx, int p_idx, double x_val, double y_val): camera_index(c_idx), point_index(p_idx), x(x_val), y(y_val) {}
    Observation(): camera_index(0), point_index(0), x(0), y(0) {}
};

struct Camera {
    Eigen::Vector3d R;
    Eigen::Vector3d t;
    double f;
    double k1;
    double k2;
};

struct BA_problem {
    int num_cameras;
    int num_points;
    int num_observations;
    bool dynamic_K;
    Observation* observations;
    Camera* cameras;
    Eigen::Vector3d* points;

    BA_problem() : num_cameras(0), num_points(0), num_observations(0), dynamic_K(false), observations(NULL), cameras(NULL), points(NULL){}

    ~BA_problem(){
        if (observations != NULL) delete[] observations;
        if (cameras != NULL) delete[] cameras;
        if (points != NULL) delete[] points;
    }
};