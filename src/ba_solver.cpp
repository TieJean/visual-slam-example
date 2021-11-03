#include <string>
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ba_solver.h"

using namespace std;
using namespace cv;
using std::pair;
using Eigen::Matrix;

namespace ba_solver {

BASolver::BASolver() :
    img_dir("../../data/gray/"),
    depth_dir("../../data/depth/") {
    }

void jocobian(Mat* J_ptr) {
    Mat& J = *J_ptr;

}
// min ||(img - depth * P * X)^2||
// 
void BASolver::solve(const vector<string>& ts) {
    vector<Mat> imgs;
    vector<Mat> depths;
    for (string t : ts) {
        imgs.push_back(imread(img_dir + t + ".png", 0));
        depths.push_back(imread(depth_dir + t + ".png", 0));
    }
    N_cameras = ts.size();

}

}

int main() {
    ba_solver::BASolver solver;
    vector<string> ts;
    ts.push_back("1311877812.989574");
    solver.solve(ts);
    return 0;
}