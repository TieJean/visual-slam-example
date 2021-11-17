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
    inputs.clear();
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
            SolverInput input, input_curr;
            input.setPose(poses[t]);
            input.setKeyPoint(kp[match.trainIdx]);
            input_curr.setPose(poses[T]);
            input_curr.setKeyPoint(kp[match.trainIdx]);
            idx = inputsFind_(input);
            if (idx != -1) {
                input_curr.setLandmark(inputs[idx].landmark);
            } else {
                x = kp[match.trainIdx].pt.x;
                y = kp[match.trainIdx].pt.y;
                imgToWorld_(x, y, depth, &X, &Y, &Z);
                input.setLandmark(X, Y, Z);
                input_curr.setLandmark(X, Y, Z);
                inputs.push_back(input);
            }
            inputs.push_back(input_curr);
        }
    }
}

void Slam::observeOdometry(const Vector3f& odom_loc ,const Quaternionf& odom_angle) {
    // if ( poses.size() != 0 || getDist_(prev_odom_loc_, odom_loc) > MIN_DELTA_D || getQuaternionDelta_(prev_odom_angle_, odom_angle).norm() < MIN_DELTA_A ) { 
        prev_odom_loc_ = odom_loc;
        prev_odom_angle_ = odom_angle;
        poses.emplace_back(prev_odom_loc_, prev_odom_angle_);
        has_new_pose_ = true; 
    // }
}

vector<pair<Vector3f, Quaternionf>>& Slam::getPoses() { return poses; }

void Slam::optimize(bool minimizer_progress_to_stdout, bool briefReport, bool fullReport) {
    Problem problem;
    for (size_t i = 0; i < inputs.size(); ++i) {
        CostFunction* cost_function = ReprojectionError::Create(inputs[i].measurement[0], inputs[i].measurement[1]);
        problem.AddResidualBlock(cost_function, NULL, inputs[i].pose, inputs[i].landmark);
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = minimizer_progress_to_stdout;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (briefReport) std::cout << summary.FullReport() << "\n";
    if (fullReport)  std::cout << summary.BriefReport() << "\n";
}

void Slam::displayInputs() {
    printf("-------display inputs---------\n");
    for (size_t i = 0; i < inputs.size(); ++i) {
        printf("%ld ", i);
        printf("camera: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            inputs[i].pose[0], inputs[i].pose[1], inputs[i].pose[2], inputs[i].pose[3],
            inputs[i].pose[4], inputs[i].pose[5], inputs[i].pose[6]);
        printf("landmark: %.2f | %.2f | %.2f \n", inputs[i].landmark[0], inputs[i].landmark[1], inputs[i].landmark[2]);
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
    vector<double*> xs;
    vector<double*> ls;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!vectorContains_(xs, inputs[i].pose, 7)) {
            xs.push_back(inputs[i].pose);
        }
        if (!vectorContains_(ls, inputs[i].landmark, 3)) {
            ls.push_back(inputs[i].landmark);
        }
    }
    printf("----------poses----------\n");
    for (size_t i = 0; i < xs.size(); ++i) {
        printf("%ld: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            i, xs[i][0], xs[i][1], xs[i][2], xs[i][3], xs[i][4], xs[i][5], xs[i][6]);
    }
    printf("----------landmarks----------\n");
    for (size_t i = 0; i < ls.size(); ++i) {
        printf("%ld: %.2f | %.2f | %.2f \n", i, ls[i][0], ls[i][1], ls[i][2]);
    }
}


int Slam::inputsFind_(const SolverInput& input) {
    for (int i = 0; i < inputs.size(); ++i) {
        if (input == inputs[i]) {return i;}
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

void Slam::imgToWorld_(const size_t& x, const size_t& y, const Mat& depth,
                 float* X_ptr, float* Y_ptr, float* Z_ptr) {
    float &X = *X_ptr;
    float &Y = *Y_ptr;
    float &Z = *Z_ptr;

    float factor = 5000.0;

    Z = (depth.at<ushort>(y, x)) / factor;
    X = (x - cx) * Z / fx;
    Y = (Y - cy) * Z / fy;
}



}
