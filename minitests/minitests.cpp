#include <cp-neural.h>

// Manual build:
// g++ -g -ggdb -I ../cpneural -I /usr/local/include/eigen3 minitest.cpp -L ../Build/cpneural/ -lcpneural -lpthread -o test

using std::cerr; using std::endl;

bool matComp(MatrixN& m0, MatrixN& m1, string msg="", floatN eps=1.e-6) {
    if (m0.cols() != m1.cols() || m0.rows() != m1.rows()) {
        cerr << msg << ": Incompatible shapes " << shape(m0) << "!=" << shape(m1) << endl;
        return false;
    }
    MatrixN d = m0 - m1;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {
        cerr << msg << " err=" << dif << endl;
        return true;
    } else {
        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        cerr << msg << " m0:" << endl << m0.format(CleanFmt) << endl;
        cerr << msg << " m1:" << endl << m1.format(CleanFmt) << endl;
        cerr << msg << "  ∂:" << endl << (m0-m1).format(CleanFmt) << endl;
        cerr << "err=" << dif << endl;
        return false;
    }
}

int doTests() {
    MatrixN w(10,10);
    MatrixN wx;
    cerr << "-------------Standard-init" << endl;
    wx=xavierInit(w,XavierMode::XAV_STANDARD);
    cerr << wx << endl;
    cerr << "-------------Normal-init" << endl;
    wx=xavierInit(w,XavierMode::XAV_NORMAL);
    cerr << wx << endl;
    cerr << "-------------Orthonormal-init" << endl;
    wx=xavierInit(w,XavierMode::XAV_ORTHONORMAL);
    cerr << wx << endl;

    cerr << "-------------Orthonormal-init (check multi)" << endl;
    MatrixN o=wx * wx.transpose();
    cerr << o << endl;

    return 0;
}

int main(int argc, char *argv[]) {
    string name="test";
    cpInitCompute(name);
    int ret=0;
    ret=doTests();
    cpExitCompute();
    return ret;
}
