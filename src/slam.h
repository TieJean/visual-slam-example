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
const float MIN_DELTA_D = 0.5;
const float MIN_DELTA_A = 0.8;
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

const size_t poseDim = 7;
const size_t landmarkDim = 3;
const size_t measurementDim = 2;

// camera, landmark, measurement idx
class CLM {
public:
    size_t poseIdx;
    size_t landmarkIdx;
    double* measurement;

    CLM() {
        measurement = new double[measurementDim];
    }

    ~CLM() {

    }

    void setPoseIdx(size_t poseIdx) {
        this->poseIdx = poseIdx;
    }

    void setLandmarkIdx(size_t landmarkIdx) {
        this->landmarkIdx = landmarkIdx;
    }

    void setMeasurement(KeyPoint kp) {
        this->measurement[0] = kp.pt.x;
        this->measurement[1] = kp.pt.y;
    }

    void setMeasurement(double* measurement) {
        for (size_t i = 0; i < measurementDim; ++i) {
            this->measurement[i] = measurement[i];
        }
    }

    bool operator==(const CLM& other) const {
        return poseIdx == other.poseIdx 
            && measurement[0] == other.measurement[0]
            && measurement[1] == other.measurement[1];
    }
};

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
    bool optimize(bool minimizer_progress_to_stdout, bool briefReport, bool fullReport);
    void displayCLMS();
    void displayPosesAndLandmarkcs();

private:
    FeatureTracker feature_tracker;
    Vector3f prev_odom_loc_;
    Quaternionf prev_odom_angle_;
    bool has_new_pose_;
    size_t t_start;
    vector<double*> poses;
    vector<double*> landmarks;
    vector<CLM> clms;
    vector<pair<vector<KeyPoint>, Mat>> features;

    int clmsFind_(const CLM& clm);
    Quaternionf getQuaternionDelta_(const Quaternionf& a1, const Quaternionf& a2);
    float getDist_(const Vector3f& odom1, const Vector3f& odom2);
    void imgToWorld_(const size_t& u, const size_t& v, const Mat& depth,
                     float* X_ptr, float* Y_ptr, float* Z_ptr);

};

} 

#endif