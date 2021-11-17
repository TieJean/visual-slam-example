#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"

#include "slam.h"


using namespace std;
using namespace ceres;

namespace slam {

Slam::Slam() :
    feature_tracker(),
    prev_odom_loc_(0.0,0.0,0.0),
    prev_odom_angle_(0.0,0.0,0.0,0.0),
    has_new_pose_(false) {
    }

void Slam::init() {
    poses.clear();
    landmarks.clear();
    clms.clear();
    features.clear();
}

void Slam::observeImage(const Mat& img, const Mat& depth) {
    if (!has_new_pose_) { return; }
    has_new_pose_ = false;

    vector<KeyPoint> kp_curr;
    Mat des_curr;
    feature_tracker.getKpAndDes(img, &kp_curr, &des_curr);
    features.emplace_back(kp_curr, des_curr);

    if (features.size() < 2) { return; }

    vector<KeyPoint> kp;
    Mat des;
    vector<DMatch> matches;

    int idx, x, y;
    float X, Y, Z;
    size_t T = features.size() - 1;
    for (size_t t = 0; t < T; ++t) {
        kp = features[t].first;
        des = features[t].second;
        feature_tracker.match(des_curr, des, &matches);
        for (auto match : matches) {
            CLM clm, clm_curr;
            clm.setPoseIdx(t);
            clm.setMeasurement(kp[match.trainIdx]);
            clm_curr.setPoseIdx(T);
            clm_curr.setMeasurement(kp[match.queryIdx]);
            idx = clmsFind_(clm); // TODO: don't use hash for debugging; need to change back to hash
            if (idx != -1) {
                clm_curr.setLandmarkIdx(clms[idx].landmarkIdx);
            } else {
                x = kp[match.trainIdx].pt.x;
                y = kp[match.trainIdx].pt.y;
                imgToWorld_(poses[t], x, y, depth, &X, &Y, &Z);
                clm.setLandmarkIdx(landmarks.size());
                clms.push_back(clm);
                clm_curr.setLandmarkIdx(landmarks.size());
                double* landmark = new double[]{X, Y, Z};
                landmarks.push_back(landmark);
            }
            clms.push_back(clm_curr);
        }
    }
}

void Slam::observeOdometry(const Vector3f& odom_loc ,const Quaternionf& odom_angle) {
    // if ( poses.size() != 0 || getDist_(prev_odom_loc_, odom_loc) > MIN_DELTA_D || getQuaternionDelta_(prev_odom_angle_, odom_angle).norm() < MIN_DELTA_A ) { 
        prev_odom_loc_ = odom_loc;
        prev_odom_angle_ = odom_angle;
        double* pose = new double[]{odom_loc.x(), odom_loc.y(), odom_loc.z(), 
                         odom_angle.w(), odom_angle.x(), odom_angle.y(), odom_angle.z()};
        poses.push_back(pose);
        has_new_pose_ = true; 
    // }
}

bool Slam::optimize(bool minimizer_progress_to_stdout, bool briefReport, bool fullReport) {
    Problem problem;
    for (size_t i = 0; i < clms.size(); ++i) {
        CostFunction* cost_function = ReprojectionError::Create(clms[i].measurement[0], clms[i].measurement[1]);
        problem.AddResidualBlock(cost_function, NULL, poses[clms[i].poseIdx], landmarks[clms[i].landmarkIdx]);
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = minimizer_progress_to_stdout;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (briefReport) std::cout << summary.FullReport() << "\n";
    if (fullReport)  std::cout << summary.BriefReport() << "\n";
    return (summary.termination_type == ceres::CONVERGENCE 
         || summary.termination_type == ceres::USER_SUCCESS);
}

void Slam::displayCLMS() {
    printf("-------display inputs---------\n");
    for (size_t i = 0; i < clms.size(); ++i) {
        printf("%ld: %ld | %ld\n", i, clms[i].poseIdx, clms[i].landmarkIdx);
        printf("camera: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            poses[clms[i].poseIdx][0], poses[clms[i].poseIdx][1], poses[clms[i].poseIdx][2], poses[clms[i].poseIdx][3],
            poses[clms[i].poseIdx][4], poses[clms[i].poseIdx][5], poses[clms[i].poseIdx][6]);
        printf("landmark: %.2f | %.2f | %.2f \n", landmarks[clms[i].landmarkIdx][0], landmarks[clms[i].landmarkIdx][1], landmarks[clms[i].landmarkIdx][2]);
    }
}

bool vectorContains_(const vector<double*>& vec, double* elt, size_t size) {
    for (auto v : vec) {
        bool ret = true;
        for (size_t i = 0; i < size; ++i) {
            ret &= (v[i] == elt[i]);
        }
        if (ret) {return true;}
    }
    return false;
}

void Slam::displayPosesAndLandmarkcs() {
    printf("----------poses----------\n");
    for (size_t i = 0; i < poses.size(); ++i) {
        printf("%ld: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            i, poses[i][0], poses[i][1], poses[i][2], poses[i][3], poses[i][4], poses[i][5], poses[i][6]);
    }
    printf("----------landmarks----------\n");
    for (size_t i = 0; i < landmarks.size(); ++i) {
        printf("%ld: %.2f | %.2f | %.2f \n", i, landmarks[i][0], landmarks[i][1], landmarks[i][2]);
    }
}


int Slam::clmsFind_(const CLM& clm) {
    for (int i = 0; i < clms.size(); ++i) {
        if (clm == clms[i]) {return i;}
    }
    return -1;
}

Quaternionf Slam::getQuaternionDelta_(const Quaternionf& a1, const Quaternionf& a2) {
    return a2 * a1;
}

float Slam::getDist_(const Vector3f& odom1, const Vector3f& odom2) {
    float x_pow = pow(odom1.x()-odom2.x(), 2);
    float y_pow = pow(odom1.y()-odom2.y(), 2);
    float z_pow = pow(odom1.z()-odom2.z(), 2);
    return sqrt( x_pow + y_pow + z_pow);
}

void Slam::imgToWorld_(double* camera, const size_t& x, const size_t& y, const Mat& depth,
                 float* X_ptr, float* Y_ptr, float* Z_ptr) {
    float &X = *X_ptr;
    float &Y = *Y_ptr;
    float &Z = *Z_ptr;

    const float factor = 5000.0;

    Z = (depth.at<ushort>(y, x)) / factor;
    X = (x - cx) * Z / fx;
    Y = (Y - cy) * Z / fy;
    Quaternionf r(camera[0], camera[1], camera[2], camera[3]);
    Vector3f landmark_trans = r.inverse() * Vector3f(X - camera[4], Y - camera[5], Z - camera[6]);
    X = landmark_trans.x();
    Y = landmark_trans.y();
    Z = landmark_trans.z();
}



}
