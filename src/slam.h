#include <vector>
#include <utility>
#include <unordered_map>
#include <cmath>
#include <array>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"

#include "feature_tracker.h"

using namespace std;
using namespace ceres;
using namespace Eigen;
using namespace cv;
using namespace feature_tracker;

#ifndef SLAM_H_
#define SLAM_H_

namespace slam {

const float kEpsilon = 1e-4;

class SolverInput {
public:
    double* pose;
    double* landmark;
    double* measurement;
    
    SolverInput() {
        pose = new double[7];
        landmark = new double[3];
        measurement = new double[2];
    }
    ~SolverInput() {
        // if (pose != nullptr) {delete pose;}
        // if (landmark != nullptr) {delete landmark;}
        // if (measurement != nullptr) {delete measurement;}
    }

    bool operator==(const SolverInput& other) const {
        bool ret = true;
        for (size_t i = 0; i < 7; ++i) {
            if (this->pose[i] != other.pose[i]) {return false;}
        }
        for (size_t i = 0; i < 2; ++i) {
            if (this->measurement[i] != other.measurement[i]) {return false;}
        }
    }

    void setPose(const pair<Vector3f, Quaternionf>& pose) {
        // if (this->pose == nullptr) {this->pose = new double[7];}
        this->pose[0] = pose.first.x();
        this->pose[1] = pose.first.y();
        this->pose[2] = pose.first.z();
        this->pose[3] = pose.second.w();
        this->pose[4] = pose.second.x();
        this->pose[5] = pose.second.y();
        this->pose[6] = pose.second.z();
    }

    void setLandmark(double* landmark) {
        this->landmark[0] = landmark[0];
        this->landmark[1] = landmark[1];
        this->landmark[2] = landmark[2];
    }

    void setLandmark(const Vector3f& landmark) {
        this->landmark[0] = landmark.x();
        this->landmark[1] = landmark.y();
        this->landmark[2] = landmark.z();
    }

    void setLandmark(const float& X, const float& Y, const float& Z) {
        this->landmark[0] = X;
        this->landmark[1] = Y;
        this->landmark[2] = Z;
    }

    void setKeyPoint(const KeyPoint& kp) {
        this->measurement[0] = kp.pt.x;
        this->measurement[1] = kp.pt.y;
    }
};

const float fx = 520.9;
const float fy = 521.0;
const float cx = 325.1;
const float cy = 249.7;
const float d0 = 0.2312;
const float d1 = -0.7849;
const float d2 = -0.0033;
const float d3 = -0.0001;
const float d4 = 0.9172;
const float ds = 1.031;

struct ReprojectionError {
    float observed_x;
    float observed_y;

    ReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}
    
    // camera - 0,1,2,3 angle-axis rotation
    //        - 4,5,6 translation
    template <typename T>
    bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(camera, point, p); // rotation
        p[0] += camera[4]; p[1] += camera[5]; p[2] += camera[6]; // translation
        
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];
        T predicted_x = - xp * T(fx) + T(cx);
        T predicted_y = - yp * T(fy) + T(cy);
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x,
                                      const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>(
                    new ReprojectionError(observed_x, observed_y)));
    }

};

class Slam {

public: 
    Slam();

    void init();
    void observeImage(const Mat& img, const Mat& depth);
    void observeOdometry(const Vector3f& odom_loc ,const Quaternionf& odom_angle);
    void optimize(bool minimizer_progress_to_stdout, bool briefReport, bool fullReport);
    vector<pair<Vector3f, Quaternionf>>& getPoses();
    vector<Vector3f>& getLandmarks();
    void displayInputs();
    void displayPosesAndLandmarkcs();

private:
    FeatureTracker feature_tracker;
    Vector3f prev_odom_loc_;
    Quaternionf prev_odom_angle_;
    bool has_new_pose_;
    vector<pair<Vector3f, Quaternionf>> poses;
    vector<SolverInput> inputs;
    vector<pair<vector<KeyPoint>, Mat>> features;

    float MIN_DELTA_D = 0.5;
    float MIN_DELTA_A = 0.8;

    int inputsFind_(const SolverInput& input);
    Quaternionf getQuaternionDelta_(const Quaternionf& a1, const Quaternionf& a2);
    float getDist_(const Vector3f& odom1, const Vector3f& odom2);
    void imgToWorld_(const size_t& u, const size_t& v, const Mat& depth,
                     float* X_ptr, float* Y_ptr, float* Z_ptr);

};

} 

#endif