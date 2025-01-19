#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

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