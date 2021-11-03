#include <string>

using namespace std;

#ifndef BUNDLE_H_
#define BUNDLE_H_

namespace ba_solver {

class BASolver {

public:
    BASolver();

    void solve(const vector<string>& ts);

private:
    string img_dir;
    string depth_dir;
    int N_cameras;

};

}

#endif // BUNDLE_H_