#pragma once

#include <iostream>
#include <fstream>
#include "Eigen"

using namespace std;
using namespace Eigen;

/*** 
 * Load data of the Bundle Adjustment in the Large Dataset 
 * Format:
 * <num_cameras> <num_points> <num_observations>
 * <camera_index_1> <point_index_1> <x_1> <y_1>
 * ...
 * <camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
 * <camera_1>
 * ...
 * <camera_num_cameras>
 * <point_1>
 * ...
 * <point_num_points>
 */

struct observation {
    int camera_index;
    int point_index;
    double x;
    double y;
};

struct camera {
    Vector3d R;
    Vector3d t;
    double f;
    double k1;
    double k2;
};

struct bal_problem {
    int num_cameras;
    int num_points;
    int num_observations;
    observation* observations;
    camera* cameras;
    Vector3d* points;

    bal_problem():num_cameras(0), num_points(0), num_observations(0), observations(NULL), cameras(NULL), points(NULL){}

    ~bal_problem(){
        if (observations) delete[] observations;
        if (cameras) delete[] cameras;
        if (points) delete[] points;
    }
};

bool load_bal(const string& path, bal_problem& p){
    ifstream infile(path);

    if (!infile.is_open()) return false;

    infile >> p.num_cameras >> p.num_points >> p.num_observations;

    p.cameras = new camera[p.num_cameras];
    p.points = new Vector3d[p.num_points];
    p.observations = new observation[p.num_observations];

    for (int i = 0; i < p.num_observations; i++){
        auto& obs = p.observations[i];
        infile >> obs.camera_index >> obs.point_index >> obs.x >> obs.y;
    }

    for (int i = 0; i < p.num_cameras; i++){
        auto& camera = p.cameras[i];
        infile >> camera.R.x() >> p.cameras[i].R.y() >> camera.R.z() 
            >> camera.t.x() >> camera.t.y() >> camera.t.z()
            >> p.cameras[i].f
            >> p.cameras[i].k1
            >> p.cameras[i].k2;
    }

    for (int i = 0; i < p.num_points; i++){
        auto& point  = p.points[i];
        infile >> point[0] >> point[1] >> point[2];
    }

    infile.close();

    return true;
}
