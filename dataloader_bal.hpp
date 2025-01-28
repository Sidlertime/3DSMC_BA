#pragma once

#include <iostream>
#include <fstream>
#include "Eigen"

#include "ba_types.hpp"

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

bool load_bal(const string& path, BA_problem& p){
    ifstream infile(path);

    if (!infile.is_open()) return false;

    p.dynamic_K = true;

    infile >> p.num_cameras >> p.num_points >> p.num_observations;

    if (p.cameras == NULL) delete[] p.cameras;
    p.cameras = new Camera[p.num_cameras];

    if (p.points == NULL) delete[] p.points;
    p.points = new Eigen::Vector3d[p.num_points];

    if (p.observations == NULL) delete[] p.observations;
    p.observations = new Observation[p.num_observations];

    for (int i = 0; i < p.num_observations; i++){
        auto& obs = p.observations[i];
        infile >> obs.camera_index >> obs.point_index >> obs.x >> obs.y;
    }

    for (int i = 0; i < p.num_cameras; i++){
        auto& camera = p.cameras[i];
        infile >> camera.R.x() >> p.cameras[i].R.y() >> camera.R.z() 
            >> camera.t.x() >> camera.t.y() >> camera.t.z()
            >> camera.f
            >> camera.k1
            >> camera.k2;
    }

    for (int i = 0; i < p.num_points; i++){
        auto& point  = p.points[i];
        infile >> point[0] >> point[1] >> point[2];
    }

    infile.close();

    return true;
}
