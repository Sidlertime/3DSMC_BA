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
    double cx;
    double cy;
};

struct BA_problem {
    int num_cameras;
    int num_points;
    int num_observations;
    bool dynamic_K;
    Observation* observations;
    Camera* cameras;
    Eigen::Vector3d* points;
    bool* invalid_points;
    Eigen::Vector3i* colors;

    BA_problem() : num_cameras(0), num_points(0), num_observations(0), dynamic_K(false), 
        observations(NULL), cameras(NULL), points(NULL), invalid_points(NULL), colors(NULL) {}

    ~BA_problem(){
        if (observations != NULL) delete[] observations;
        if (cameras != NULL) delete[] cameras;
        if (points != NULL) delete[] points;
        if (invalid_points != NULL) delete[] invalid_points;
        if (colors != NULL) delete[] colors;
    }
};

int BARemoveOutliersAbsolut(BA_problem& problem, double distance = 1000.0){
    if (problem.points == NULL || problem.invalid_points == NULL) return 0;

    Eigen::Vector3d center(0, 0, 0);
    for (int i = 0; i < problem.num_points; i++){
        center += problem.points[i] / problem.num_points;
    }

    int removed = 0;

    for (int i = 0; i < problem.num_points; i++){
        if ((problem.points[i] - center).norm() > distance){
            problem.invalid_points[i] = true;
            removed++;
        } else {
            problem.invalid_points[i] = false;
        }
    }

    return removed;
}

int BARemoveOutliersRelativ(BA_problem& problem, double deviation_factor = 2.0){
    if (problem.points == NULL || problem.invalid_points == NULL) return 0;

    Eigen::Vector3d center(0, 0, 0);
    for (int i = 0; i < problem.num_points; i++){
        center += problem.points[i] / problem.num_points;
    }

    double mean_distance = 0;
    for (int i = 0; i < problem.num_points; i++){
        mean_distance += (center - problem.points[i]).norm() / problem.num_points;
    }

    int removed = 0;

    for (int i = 0; i < problem.num_points; i++){
        if ((problem.points[i] - center).norm() > deviation_factor * mean_distance){
            problem.invalid_points[i] = true;
            removed++;
        } else {
            problem.invalid_points[i] = false;
        }
    }

    return removed;
}
