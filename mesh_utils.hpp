#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "Eigen"

#include "dataloader_bal.hpp"

using namespace std;
using namespace cv;

struct Vertex {
    Point3f pos;
    Vec3b color;

    Vertex(Point3f p, Vec3b c):pos(p),color(c){}
};

bool vertexToMesh(vector<Vertex> vertices, const string& filename = "mesh_out.off"){
    ofstream outFile(filename);
    if(!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << endl;

	outFile << "# numVertices numFaces numEdges" << endl;

	outFile << vertices.size() << " 0 0" << endl;

    for (auto& v : vertices){
        if(isfinite(v.pos.x + v.pos.y + v.pos.z)){
            // coordinates and color
            outFile << v.pos.x << " " << v.pos.y << " " << v.pos.z << " " << int(v.color[0]) << " " << int(v.color[1]) << " " << int(v.color[2]) << " 0"<< endl;
        } else {
            outFile << "0.0 0.0 0.0 0 0 0 0" << endl;
        }
    }

    outFile.close();

    return true;
}


bool pointsToMesh(vector<Point3f> points, vector<Point3f> cameraPositions, const string& filename = "mesh_out.off", float scale = 1.0){
    ofstream outFile(filename);
    if(!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << endl;

	outFile << "# numVertices numFaces numEdges" << endl;

	outFile << points.size() + cameraPositions.size() << " 0 0" << endl;

    for(auto& c : cameraPositions){
        if(isfinite(c.x + c.y + c.z)){
            // coordinates and color Red
            outFile << scale * c.x << " " << scale * c.y << " " << scale * c.z << " 255 0 0 0" << endl;
        } else {
            outFile << "0.0 0.0 0.0 0 0 0 0" << endl;
        }
    }

    for (auto& p : points){
        if(isfinite(p.x + p.y + p.z)){
            // coordinates and color Black
            outFile << scale * p.x << " " << scale * p.y << " " << scale * p.z << " 0 0 0 0" << endl;
        } else {
            outFile << "0.0 0.0 0.0 0 0 0 0" << endl;
        }
    }

    outFile.close();

    return true;
}

bool balToMesh(BA_problem& bal, const string& filename = "mesh_out.off", bool draw_cams = true, bool draw_cam_direction = true){
    cout << "Writing mesh " << filename << endl;

    ofstream outFile(filename);
    if(!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << endl;

	outFile << "# numVertices numFaces numEdges" << endl;

	outFile << bal.num_points + (draw_cams ? bal.num_cameras : 0) + (draw_cam_direction ? bal.num_cameras : 0) << " 0 0" << endl;

    for (int i = 0; i < bal.num_points; i++){
        auto& p = bal.points[i];
        Eigen::Vector3i c(0, 0, 0);
        if (bal.colors != NULL) c = bal.colors[i];
        if(p.allFinite()){
            // Point coordinates in color Black
            outFile << p.x() << " " << p.y() << " " << p.z() << " " << c.x() << " " << c.y() << " " << c.z() << " 0" << endl;
        } else {
            outFile << "0.0 0.0 0.0 0 0 0 0" << endl;
        }
    }

    for (int i = 0; i < bal.num_cameras; i++){
        auto& c = bal.cameras[i];
        Eigen::Matrix3d rot;
        ceres::AngleAxisToRotationMatrix(c.R.data(), rot.data());
        Eigen::Vector3d origin = - (rot.transpose() * c.t);

        if (draw_cams){
            if(origin.allFinite()){
                // Camera coordinates in color Red
                outFile << origin.x() << " " << origin.y() << " " << origin.z() << " 255 0 0 0" << endl;
            } else {
                outFile << "0.0 0.0 0.0 0 0 0 0" << endl;
            }
        }

        // also add a point 1 unit infront of the cameras direction
        if (draw_cam_direction){
            Eigen::Vector3d unit(0, 0, 1);
            unit = rot.transpose() * (unit - c.t);
            if(unit.allFinite()){
                // Camera coordinates in color Red
                outFile << unit.x() << " " << unit.y() << " " << unit.z() << " 0 255 0 0" << endl;
            } else {
                outFile << "0.0 0.0 0.0 0 0 0 0" << endl;
            }
        }
    }

    cout << "Finished writing BAL to mesh " << filename << endl;

    outFile.close();

    return true;
}
