#include <vector>

#include "feature_tracker.h"
#include "slam.h"

using namespace std;
using namespace gtsam;
using namespace slam;

int main() {
    
    vector<string> timestamps;
    timestamps.push_back("1311877813.025527");
    timestamps.push_back("1311877817.026998");
    vector<Pose3> odometries;
    odometries.push_back( Pose3(Rot3::Quaternion(0.1381, 0.6691, -0.630,6 -0.1461), Point3(-2.2827, -2.496, 0.6646)) ); // 1311877813.0011
    odometries.push_back( Pose3(Rot3::Quaternion(0.3752, 0.62, -0.601, -0.3953), Point3(-2.2483, -2.3296, 0.6062)) ); // 1311877817.0675 

    Slam slam_solver;
    slam_solver.init();

    slam_solver.observeOdometry(odometries[0]);
    slam_solver.observeImage(timestamps[0]);
    slam_solver.observeOdometry(odometries[1]);
    slam_solver.observeImage(timestamps[1]);
    slam_solver.optimize();
    return 0;
}