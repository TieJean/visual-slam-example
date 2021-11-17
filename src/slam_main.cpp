#include <vector>

#include "feature_tracker.h"
#include "slam.h"

using namespace std;
using namespace slam;

int main() {
    
    vector<pair<string, string>> timestamps;
    vector<pair<Vector3f, Quaternionf>> odometries;
    timestamps.emplace_back("1311877812.989574", "1311877812.987032");
    timestamps.emplace_back("1311877815.025449", "1311877815.025457");
    timestamps.emplace_back("1311877817.026998", "1311877817.027055");
    odometries.emplace_back(Vector3f(-2.2581, -2.3799, 0.5899), Quaternionf(-0.1617, 0.1914, 0.7173, -0.6502)); 
    odometries.emplace_back(Vector3f(-2.2640, -2.3775, 0.5895), Quaternionf(-0.1929, 0.2276, 0.7059, -0.6424)); 
    odometries.emplace_back(Vector3f(-2.2869, -2.3742, 0.5884), Quaternionf(-0.2905, 0.3374, 0.6641, -0.6005));  

    Slam slam;
    slam.init();
    for (size_t i = 0; i < timestamps.size(); ++i) {
        slam.observeOdometry(odometries[i].first, odometries[i].second);
        Mat img = imread("../data/rgb/" + timestamps[i].first + ".png", 0);
        Mat depth = imread("../data/depth/" + timestamps[i].second + ".png", IMREAD_UNCHANGED);
        if (img.data == NULL || depth.data == NULL) {
            cout << "../data/rgb/" + timestamps[i].first + ".png" << endl;
            cout << "../data/depth/" + timestamps[i].second + ".png" << endl;
            perror("cannot open image files");
            exit(1);
        }
        slam.observeImage(img, depth);
    }

    // slam.displayInputs();
    // slam.displayPosesAndLandmarkcs();
    slam.optimize(false, true, false);
    // slam.displayPosesAndLandmarkcs();
    
}