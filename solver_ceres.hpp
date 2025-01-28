#pragma once

#include <optional>

#include "ceres/ceres.h"
#include "Eigen"

#include "dataloader_bal.hpp"
#include "dataloader_bf.hpp"
#include "mesh_utils.hpp"

struct ReprojectionErrorDynK {
    ReprojectionErrorDynK(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const R, const T* t, const T* const point, const T* const f, const T* const k1, const T* const k2, T* residuals) const {
        // Conversion from world to camera coordinates
        T p[3];
        ceres::AngleAxisRotatePoint(R, point, p);
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];

        // Perspective division
        T x_image = p[0] / p[2];
        T y_image = p[1] / p[2];

        // Conversion to pixel coordinates
        T sqrnorm = x_image * x_image + y_image * y_image;
        T r_p = T(1.0) + k1[0] * sqrnorm + k2[0] * sqrnorm * sqrnorm;
        x_image = f[0] * r_p * x_image;
        y_image = f[0] * r_p * y_image;

        // Compute residuals
        residuals[0] = x_image - T(observed_x);
        residuals[1] = y_image - T(observed_y);

        return true;
    }

private:
    double observed_x, observed_y;
};

struct ReprojectionErrorFixK {
    ReprojectionErrorFixK(double observed_x, double observed_y, double f, double k1, double k2)
        : observed_x(observed_x), observed_y(observed_y), f(f), k1(k1), k2(k2) {}

    template <typename T>
    bool operator()(const T* const R, const T* t, const T* const point, T* residuals) const {
        // Conversion from world to camera coordinates
        T p[3];
        ceres::AngleAxisRotatePoint(R, point, p);
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];

        // Perspective division
        p[0] = p[0] / p[2];
        p[1] = p[1] / p[2];
        p[2] = T(1.0);

        // Conversion to pixel coordinates
        T x_image = T(f) * p[0] + T(k1);
        T y_image = T(f) * p[1] + T(k2);

        // Compute residuals
        residuals[0] = x_image - T(observed_x);
        residuals[1] = y_image - T(observed_y);

        return true;
    }

private:
    double observed_x, observed_y, f, k1, k2;
};

void solveBA(BA_problem& bal){
    ceres::Problem problem;

    if(bal.dynamic_K){
        for (size_t i = 0; i < bal.num_observations; i++) {
            auto& obs = bal.observations[i];
            if(bal.invalid_points[obs.point_index]) continue;   // outlier
            auto& cam = bal.cameras[obs.camera_index];
            double* r = cam.R.data();
            double* t = cam.t.data();
            double* point = bal.points[obs.point_index].data();

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionErrorDynK, 2, 3, 3, 3, 1, 1, 1>(
                new ReprojectionErrorDynK(obs.x, obs.y));
            problem.AddResidualBlock(cost_function, nullptr, r, t, point, &cam.f, &cam.k1, &cam.k2);
        }
    } else {
        for (size_t i = 0; i < bal.num_observations; i++) {
            auto& obs = bal.observations[i];
            //cout << "Observation: " << i << " with camera: " << obs.camera_index << " point: " << obs.point_index << " at " << obs.x << ", " << obs.y << endl;
            if(bal.invalid_points[obs.point_index]) continue;   // outlier
            if(obs.camera_index >= bal.num_cameras){
                cout << "PANIK with observation " << i << " has camera index " << obs.camera_index << endl;
            }
            auto& cam = bal.cameras[obs.camera_index];
            double* r = cam.R.data();
            double* t = cam.t.data();
            if(obs.point_index >= bal.num_points){
                cout << "PANIK with observation " << i << " has point index " << obs.point_index << endl;
            }
            double* point = bal.points[obs.point_index].data();

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionErrorFixK, 2, 3, 3, 3>(
                new ReprojectionErrorFixK(obs.x, obs.y, cam.f, cam.k1, cam.k2));
            problem.AddResidualBlock(cost_function, nullptr, r, t, point);
            //problem.SetParameterBlockConstant(r);
            //problem.SetParameterBlockConstant(t);
        }
    }

    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << endl;

    //balToMesh(bal, "bal_optimized.off");
}
