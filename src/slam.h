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
const float fx = 400.000000; 
const float fy = 400.000000; 
const float cx = 320.000000; 
const float cy = 240.000000;

const size_t poseDim = 7;
const size_t landmarkDim = 3;
const size_t measurementDim = 2;
const size_t imgHeight = 480;
const size_t imgWidth = 640;
const int imgEdge = 50;

// camera, landmark, measurement idx
class CLM {
public:
    size_t poseIdx;
    size_t landmarkIdx;
    double* measurement;

    CLM() {
        measurement = new double[measurementDim];
    }

    ~CLM() { }

    void setPoseIdx(size_t poseIdx) { this->poseIdx = poseIdx; }

    void setLandmarkIdx(size_t landmarkIdx) { this->landmarkIdx = landmarkIdx;  }

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
        
        // p: translate world to base_link
        T p[3];
        ceres::QuaternionRotatePoint(camera, point, p);
        p[0] += camera[4]; p[1] += camera[5]; p[2] += camera[6]; // translation
        
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        T predicted_x = xp * T(fx) + T(cx);
        T predicted_y = yp * T(fy) + T(cy);
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        if (predicted_x >= T(imgWidth + imgEdge) || predicted_x < T(-imgEdge) || predicted_y >= T(imgHeight + imgEdge) || predicted_y < T(-imgEdge)) {
            // printf("landmark outside range: \n");
            residuals[0] = T(0);
            residuals[1] = T(0);
        }

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
    bool debug = false;
    Slam();

    void init();
    void observeImage(const Mat& img, const Mat& depth);
    void observeOdometry(const Vector3f& odom_loc ,const Quaternionf& odom_angle);
    bool optimize(bool minimizer_progress_to_stdout, bool briefReport, bool fullReport);
    void displayCLMS();
    void displayPoses();
    void displayLandmarks();
    void imgToWorld_(double* camera, const int& x, const int& y, const int& z,
                 float* X_ptr, float* Y_ptr, float* Z_ptr);
    void imgToWorld_(double* camera, const int& u, const int& v, const Mat& depth,
                     float* X_ptr, float* Y_ptr, float* Z_ptr);
    bool worldToImg_(double* camera, const float& X, const float& Y, const float& Z,
                     float* x_ptr, float* y_ptr);
private:
    FeatureTracker feature_tracker;
    Vector3f prev_odom_loc_;
    Quaternionf prev_odom_angle_;
    bool has_new_pose_;
    size_t t_start;
    Matrix3f cameraIntrinsic;

    vector<double*> poses;
    vector<double*> landmarks;

    vector<double*> cameras;
    vector<double*> points;
    vector<unsigned short> point_cnts;
    vector<CLM> clms;
    vector<pair<vector<KeyPoint>, Mat>> features;
    vector<Mat> depths; // TODO: optimize space

    int clmsFind_(const CLM& clm);
    Quaternionf getQuaternionDelta_(const Quaternionf& a1, const Quaternionf& a2);
    float getDist_(const Vector3f& odom1, const Vector3f& odom2);
    
    // bool wolrdToImg_(double* camera, const float& X, const float& Y, const float& Z,
    //                  float* x_ptr, float* y_ptr);
    

};

} 

#endif