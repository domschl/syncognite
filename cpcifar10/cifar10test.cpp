#include <iostream>
#include <string>
#include <H5Cpp.h>
#include <H5File.h>
#include <H5DataSet.h>

#include "cp-neural.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

typedef struct {
    int index;               // index of the current object
} cp_cifar10_iter_info;

herr_t cp_cifar10_get_all_groups(hid_t loc_id, const char *name, void *opdata);

H5::H5File *cp_cifar10_pfile;
map<string, Tensor<floatN, 4> *> cpcifar10Data4;
map<string, MatrixN *> cpcifar10Data;

herr_t cp_cifar10_get_all_groups(hid_t loc_id, const char *name, void *opdata)
{
    cp_cifar10_iter_info *info=(cp_cifar10_iter_info *)opdata;

    // Here you can do whatever with the name...
    cout << name << " ";

    // you can use this call to select just the groups
    // H5G_LINK    0  Object is a symbolic link.
    // H5G_GROUP   1  Object is a group.
    // H5G_DATASET 2  Object is a dataset.
    // H5G_TYPE    3  Object is a named datatype.
    int obj_type = H5Gget_objtype_by_idx(loc_id, info->index);
    //if(obj_type == H5G_GROUP)
	   //cout << "        is a group" << endl;
    if (obj_type == H5G_DATASET) {
        H5::DataSet dataset = cp_cifar10_pfile->openDataSet(name);
        H5::DataSpace filespace = dataset.getSpace();
        int rank = filespace.getSimpleExtentNdims();
        // cout << "rank: " << rank << endl;
        hsize_t dims[10];    // dataset dimensions
        rank = filespace.getSimpleExtentDims( dims );
        cout << "dataset rank = " << rank << ", dimensions; ";
        for (int j=0; j<rank; j++) {
            cout << (unsigned long)(dims[j]);
            if (j<rank-1) cout << " x ";
        }
        H5::DataSpace mspace1(rank, dims);
        auto dataClass = dataset.getTypeClass();
        switch (dataClass) {
            case H5T_FLOAT:
                if (dataset.getFloatType().getSize()==4) {
                    cout << " float" << endl;
                } else if (dataset.getFloatType().getSize()==8) {
                    cout << " double" << endl;
                }
                break;
            case H5T_INTEGER:
                cout << " int" << endl;
                break;
            default:
                cout << "dataClass not implemented: " << dataClass << endl;
                break;
        }

        int *pi, *pi4;
        if (rank==1) {
            rank=2;
            dims[1]=1;
        }
       switch (rank) {
            case 2:
                cpcifar10Data[name] = new MatrixN(dims[0],dims[1]);
                pi = (int *)malloc(sizeof(int) * dims[0] * dims[1]);
                if (pi) {
                    dataset.read(pi, H5::PredType::NATIVE_INT, mspace1, filespace );
                    for (int y=0; y<dims[0]; y++) {
                        for (int x=0; x<dims[1]; x++) {
                            (*(cpcifar10Data[name]))(y,x)=(floatN)pi[y*dims[1]+x];
                        }
                    }
                    free(pi);
                }
                break;
            case 4:
                cpcifar10Data4[name] = new Tensor<floatN, 4>((long int)dims[0],(long int)dims[1],(long int)dims[2],(long int)dims[3]);
                cpcifar10Data[name] = new MatrixN(dims[0],dims[1]*dims[2]*dims[3]);
                pi4 = (int *)malloc(sizeof(int) * dims[0] * dims[1] * dims[2] * dims[3]);
                if (pi4) {
                    //int doout=10000;
                    dataset.read(pi4, H5::PredType::NATIVE_INT, mspace1, filespace );
                    for (int z=0; z<dims[0]; z++) {
                        for (int y=0; y<dims[1]; y++) {
                            for (int x=0; x<dims[2]; x++) {
                                for (int w=0; w<dims[3]; w++) {
                                    (*(cpcifar10Data4[name]))(z,y,x,w)=(floatN)pi4[z*dims[3]*dims[2]*dims[1] + y*dims[3]*dims[2]+x*dims[3] + w];
                                    (*(cpcifar10Data[name]))(z,y*dims[3]*dims[2]+x*dims[3]+w)=(floatN)(pi4[z*dims[3]*dims[2]*dims[1] + y*dims[3]*dims[2]+x*dims[3] + w]) / 256.0 - 0.5;
/*                                    while (doout>0) {
                                        cout << "[" << pi4[z*dims[3]*dims[2]*dims[1] + y*dims[3]*dims[2]+x*dims[3] + w] << " -> " << (*(cpcifar10Data[name]))(z,y*dims[3]*dims[2]+x*dims[3]+w) << "]  " ;
                                        --doout;
                                    }
*/                                }
                            }
                        }
                    }
                    free(pi4);
                }
                break;
            default:
                cout << "NOT YET IMPLEMENTED RANK: " << rank << endl;
                exit(-1);
        }
    }

    (info->index)++;

return 0;
 }

bool  getcifar10Data(string filepath) {
    if (cpcifar10Data.size() > 0) {
        cout << "cpcifar10Data contains already elements, not reloading." << endl;
        return true;
    }
    H5::H5File fmn((H5std_string)filepath, H5F_ACC_RDONLY);
    cp_cifar10_pfile=&fmn;
    int nr=fmn.getNumObjs();
    //cout << nr << endl;
    cp_cifar10_iter_info info;
    info.index=0;
    cout << "Reading: ";
    fmn.iterateElems("/", NULL, cp_cifar10_get_all_groups, &info);
    cout << endl;
    return true;
}

/*CpParams autoOptimize(Layer *pLayer, CpParams& cpi, vector<string>& optiParams, MatrixN& X, MatrixN&y) {

}

floatN evalMultilayer(CpParams& cpo, MatrixN& X, MatrixN& y, MatrixN& Xv, MatrixN& yv, MatrixN& Xt, MatrixN& yt, bool evalFinal=false, bool verbose=false) {
    int N4=500, N5=200;
    CpParams cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9,cp10,cp11,cp12,cp13,cp14,cp15,cp16,cp17,cp18,cp19,cp20,cp21;
    floatN dropR=0.75;
    MultiLayer ml("{topo=[3072];name='multi1'}");
    if (verbose) cout << "LayerName for ml: " << ml.layerName << endl;

// l1     (W + 2 * pad - WW) % stride
    cp1.setPar("topo",vector<int>{3,32,32,16,5,5});
    cp1.setString("{stride=2;pad=1}");
    Convolution cv1(cp1);
    ml.addLayer("cv1",&cv1,vector<string>{"input"});
    cp2.setPar("topo",vector<int>{cv1.oTopo()[0]*cv1.oTopo()[1]*cv1.oTopo()[2]});
    Relu mrl1(cp2);
    ml.addLayer("rl1",&mrl1,vector<string>{"cv1"});
// l2
    cp3.setPar("topo",vector<int>{cv1.oTopo()[0],cv1.oTopo()[1],cv1.oTopo()[2],16,5,5});
    cp3.setString("{stride=1;pad=0}");
    Convolution cv2(cp3);
    ml.addLayer("cv2",&cv2,vector<string>{"rl1"});
    cp4.setPar("topo",vector<int>{cv2.oTopo()[0]*cv2.oTopo()[1]*cv2.oTopo()[2]});
    Relu mrl2(cp4);
    ml.addLayer("rl2",&mrl2,vector<string>{"cv2"});
// l3
    cp5.setPar("topo",vector<int>{cv2.oTopo()[0],cv2.oTopo()[1],cv2.oTopo()[2],16,3,3});
    cp5.setString("{stride=1;pad=0}");
    Convolution cv3(cp5);
    ml.addLayer("cv3",&cv3,vector<string>{"rl2"});
    cp6.setPar("topo",vector<int>{cv3.oTopo()[0]*cv3.oTopo()[1]*cv3.oTopo()[2]});
    Relu mrl3(cp6);
    ml.addLayer("rl3",&mrl3,vector<string>{"cv3"});
// l4
    cp7.setPar("topo",vector<int>{cv3.oTopo()[0],cv3.oTopo()[1],cv3.oTopo()[2],32,3,3});
    cp7.setString("{stride=1;pad=0}");
    Convolution cv4(cp7);
    ml.addLayer("cv4",&cv4,vector<string>{"rl3"});
    cp8.setPar("topo",vector<int>{cv4.oTopo()[0]*cv4.oTopo()[1]*cv4.oTopo()[2]});
    Relu mrl4(cp8);
    ml.addLayer("rl4",&mrl4,vector<string>{"cv4"});
// l5
    cp9.setPar("topo",vector<int>{cv4.oTopo()[0],cv4.oTopo()[1],cv4.oTopo()[2],64,3,3});
    cp9.setString("{stride=1;pad=0}");
    Convolution cv5(cp9);
    ml.addLayer("cv5",&cv5,vector<string>{"rl4"});
    cp10.setPar("topo",vector<int>{cv5.oTopo()[0]*cv5.oTopo()[1]*cv5.oTopo()[2]});
    Relu mrl5(cp10);
    ml.addLayer("rl5",&mrl5,vector<string>{"cv5"});
// l6
    cp11.setPar("topo",vector<int>{cv5.oTopo()[0],cv5.oTopo()[1],cv5.oTopo()[2],128,3,3});
    cp11.setString("{stride=1;pad=0}");
    Convolution cv6(cp11);
    ml.addLayer("cv6",&cv6,vector<string>{"rl5"});
    cp12.setPar("topo",vector<int>{cv6.oTopo()[0]*cv6.oTopo()[1]*cv6.oTopo()[2]});
    Relu mrl6(cp12);
    ml.addLayer("rl6",&mrl6,vector<string>{"cv6"});
// l7
    cp13.setPar("topo",vector<int>{cv6.oTopo()[0]*cv6.oTopo()[1]*cv6.oTopo()[2],N4});
    Affine maf4(cp13);
    ml.addLayer("af4",&maf4,vector<string>{"rl6"});

    cp14.setPar("topo", vector<int>{N4});
    BatchNorm bn4(cp14);
    ml.addLayer("bn4",&bn4,vector<string>{"af4"});

    cp15.setPar("topo", vector<int>{N4});
    Relu mrl7(cp15);
    ml.addLayer("rl7",&mrl7,vector<string>{"bn4"});

    cp16.setPar("topo", vector<int>{N4});
    cp16.setPar("drop", dropR);
    Dropout dr4(cp16);
    ml.addLayer("dr4",&dr4,vector<string>{"rl7"});
// l8
    cp17.setPar("topo",vector<int>{N4,N5});
    Affine maf5(cp17);
    ml.addLayer("af5",&maf5,vector<string>{"dr4"});

    cp18.setPar("topo", vector<int>{N5});
    BatchNorm bn5(cp18);
    ml.addLayer("bn5",&bn5,vector<string>{"af5"});

    cp19.setPar("topo", vector<int>{N5});
    Relu mrl8(cp19);
    ml.addLayer("rl8",&mrl8,vector<string>{"bn5"});

    cp20.setPar("topo", vector<int>{N5});
    cp20.setPar("drop", dropR);
    Dropout dr5(cp20);
    ml.addLayer("dr5",&dr5,vector<string>{"rl8"});
// l9
    cp21.setPar("topo",vector<int>{N5,10});
    Affine maf6(cp21);
    ml.addLayer("af6",&maf6,vector<string>{"dr5"});

    Softmax msm1("{topo=[10]}");
    ml.addLayer("sm1",&msm1,vector<string>{"af6"});
    if (!ml.checkTopology(verbose)) {
        if (verbose) cout << "Topology-check for MultiLayer: ERROR." << endl;
    } else {
        if (verbose) cout << "Topology-check for MultiLayer: ok." << endl;
    }

    floatN cAcc=ml.train(X, y, Xv, yv, "Adam", cpo);
    floatN final_err;

    if (evalFinal) {
        final_err=ml.test(Xt, yt);
        cout << "Final error on test-set:" << final_err << ", accuracy:" << 1.0-final_err << endl;
        cAcc=1-final_err;
    }
    return cAcc;
}
*/

/*// DomConvNet
floatN evalMultilayer(CpParams& cpo, MatrixN& X, MatrixN& y, MatrixN& Xv, MatrixN& yv, MatrixN& Xt, MatrixN& yt, bool evalFinal=false, bool verbose=false) {
    int N4=500, N5=200;
    CpParams cp1,cp2,cp3,cp4,cp41, cp42, cp5,cp6,cp7,cp8,cp9,cp10,cp11,cp12,cp13,cp14,cp15,cp16,cp17,cp18,cp19,cp20,cp21;
    floatN dropR=0.8;
    MultiLayer ml("{topo=[3072];name='multi1'}");
    if (verbose) cout << "LayerName for ml: " << ml.layerName << endl;

// l1     (W + 2 * pad - WW) % stride   ||    pad = (filter_size - 1) // 2
    cp1.setPar("topo",vector<int>{3,32,32,48,5,5});
    cp1.setString("{stride=1;pad=2}");
    Convolution cv1(cp1);
    ml.addLayer("cv1",&cv1,vector<string>{"input"});
    cp2.setPar("topo",vector<int>{cv1.oTopo()[0]*cv1.oTopo()[1]*cv1.oTopo()[2]});
    Relu mrl1(cp2);
    ml.addLayer("rl1",&mrl1,vector<string>{"cv1"});
// l2
    cp3.setPar("topo",vector<int>{cv1.oTopo()[0],cv1.oTopo()[1],cv1.oTopo()[2],48,5,5});
    cp3.setString("{stride=1;pad=2}");
    Convolution cv2(cp3);
    ml.addLayer("cv2",&cv2,vector<string>{"rl1"});
    cp4.setPar("topo",vector<int>{cv2.oTopo()[0]*cv2.oTopo()[1]*cv2.oTopo()[2]});
    Relu mrl2(cp4);
    ml.addLayer("rl2",&mrl2,vector<string>{"cv2"});
    cp41.setPar("topo",vector<int>{cv2.oTopo()[0],cv2.oTopo()[1],cv2.oTopo()[2]});
    cp41.setPar("stride",2);
    Pooling pl1(cp41);
    ml.addLayer("pl1",&pl1,vector<string>{"rl2"});

// l3
    cp5.setPar("topo",vector<int>{pl1.oTopo()[0],pl1.oTopo()[1],pl1.oTopo()[2],48,3,3});
    cp5.setString("{stride=1;pad=1}");
    Convolution cv3(cp5);
    ml.addLayer("cv3",&cv3,vector<string>{"pl1"});
    cp6.setPar("topo",vector<int>{cv3.oTopo()[0]*cv3.oTopo()[1]*cv3.oTopo()[2]});
    Relu mrl3(cp6);
    ml.addLayer("rl3",&mrl3,vector<string>{"cv3"});
// l4
    cp7.setPar("topo",vector<int>{cv3.oTopo()[0],cv3.oTopo()[1],cv3.oTopo()[2],48,3,3});
    cp7.setString("{stride=1;pad=1}");
    Convolution cv4(cp7);
    ml.addLayer("cv4",&cv4,vector<string>{"rl3"});
    cp8.setPar("topo",vector<int>{cv4.oTopo()[0]*cv4.oTopo()[1]*cv4.oTopo()[2]});
    Relu mrl4(cp8);
    ml.addLayer("rl4",&mrl4,vector<string>{"cv4"});
    cp42.setPar("topo",vector<int>{cv4.oTopo()[0],cv4.oTopo()[1],cv4.oTopo()[2]});
    cp42.setPar("stride",2);
    Pooling pl2(cp42);
    ml.addLayer("pl2",&pl2,vector<string>{"rl4"});
// // l5
//     cp9.setPar("topo",vector<int>{cv4.oTopo()[0],cv4.oTopo()[1],cv4.oTopo()[2],64,3,3});
//     cp9.setString("{stride=1;pad=0}");
//     Convolution cv5(cp9);
//     ml.addLayer("cv5",&cv5,vector<string>{"rl4"});
//     cp10.setPar("topo",vector<int>{cv5.oTopo()[0]*cv5.oTopo()[1]*cv5.oTopo()[2]});
//     Relu mrl5(cp10);
//     ml.addLayer("rl5",&mrl5,vector<string>{"cv5"});
// // l6
//     cp11.setPar("topo",vector<int>{cv5.oTopo()[0],cv5.oTopo()[1],cv5.oTopo()[2],128,3,3});
//     cp11.setString("{stride=1;pad=0}");
//     Convolution cv6(cp11);
//     ml.addLayer("cv6",&cv6,vector<string>{"rl5"});
//     cp12.setPar("topo",vector<int>{cv6.oTopo()[0]*cv6.oTopo()[1]*cv6.oTopo()[2]});
//     Relu mrl6(cp12);
//     ml.addLayer("rl6",&mrl6,vector<string>{"cv6"});
// l7
    cp13.setPar("topo",vector<int>{pl2.oTopo()[0]*pl2.oTopo()[1]*pl2.oTopo()[2],N4});
    Affine maf4(cp13);
    ml.addLayer("af4",&maf4,vector<string>{"pl2"});

    cp14.setPar("topo", vector<int>{N4});
    BatchNorm bn4(cp14);
    ml.addLayer("bn4",&bn4,vector<string>{"af4"});

    cp15.setPar("topo", vector<int>{N4});
    Relu mrl7(cp15);
    ml.addLayer("rl7",&mrl7,vector<string>{"bn4"});

    cp16.setPar("topo", vector<int>{N4});
    cp16.setPar("drop", dropR);
    Dropout dr4(cp16);
    ml.addLayer("dr4",&dr4,vector<string>{"rl7"});
// l8
    cp17.setPar("topo",vector<int>{N4,N5});
    Affine maf5(cp17);
    ml.addLayer("af5",&maf5,vector<string>{"dr4"});

    cp18.setPar("topo", vector<int>{N5});
    BatchNorm bn5(cp18);
    ml.addLayer("bn5",&bn5,vector<string>{"af5"});

    cp19.setPar("topo", vector<int>{N5});
    Relu mrl8(cp19);
    ml.addLayer("rl8",&mrl8,vector<string>{"bn5"});

    cp20.setPar("topo", vector<int>{N5});
    cp20.setPar("drop", dropR);
    Dropout dr5(cp20);
    ml.addLayer("dr5",&dr5,vector<string>{"rl8"});
// l9
    cp21.setPar("topo",vector<int>{N5,10});
    Affine maf6(cp21);
    ml.addLayer("af6",&maf6,vector<string>{"dr5"});

    Softmax msm1("{topo=[10]}");
    ml.addLayer("sm1",&msm1,vector<string>{"af6"});

    if (verbose) cout << "Checking multi-layer topology..." << endl;
    if (!ml.checkTopology(verbose)) {
        if (verbose) cout << "Topology-check for MultiLayer: ERROR." << endl;
    } else {
        if (verbose) cout << "Topology-check for MultiLayer: ok." << endl;
    }

    floatN cAcc=ml.train(X, y, Xv, yv, "Adam", cpo);
    floatN final_err;

    if (evalFinal) {
        final_err=ml.test(Xt, yt);
        cout << "Final error on test-set:" << final_err << ", accuracy:" << 1.0-final_err << endl;
        cAcc=1-final_err;
    }
    return cAcc;
}
*/
floatN evalMultilayer(CpParams& cpo, MatrixN& X, MatrixN& y, MatrixN& Xv, MatrixN& yv, MatrixN& Xt, MatrixN& yt, bool evalFinal=false, bool verbose=false) {
    int N4=500, N5=200;
    CpParams cp1,cp1x1,cp2,cp3,cp3x3,cp4,cp41, cp42, cp5,cp6,cp7,cp8,cp9,cp10,cp11,cp12,cp13,cp14,cp15,cp16,cp17,cp18,cp19,cp20,cp21;
    floatN dropR=0.5;
    MultiLayer ml("{topo=[3072];name='multi1'}");
    if (verbose) cout << "LayerName for ml: " << ml.layerName << endl;

// l1     (W + 2 * pad - WW) % stride   ||    pad = (filter_size - 1) // 2
    cp1.setPar("topo",vector<int>{3,32,32,48,5,5});
    cp1.setString("{stride=1;pad=2}");
    Convolution cv1(cp1);
    ml.addLayer("cv1",&cv1,vector<string>{"input"});

    cp1x1.setPar("topo", vector<int>{cv1.oTopo()[0]*cv1.oTopo()[1]*cv1.oTopo()[2]});
    BatchNorm bn1x1(cp1x1);
    ml.addLayer("bn1x1",&bn1x1,vector<string>{"cv1"});

    cp2.setPar("topo",vector<int>{cv1.oTopo()[0]*cv1.oTopo()[1]*cv1.oTopo()[2]});
    Relu mrl1(cp2);
    ml.addLayer("rl1",&mrl1,vector<string>{"bn1x1"});
// l2
    cp3.setPar("topo",vector<int>{cv1.oTopo()[0],cv1.oTopo()[1],cv1.oTopo()[2],48,5,5});
    cp3.setString("{stride=1;pad=2}");
    Convolution cv2(cp3);
    ml.addLayer("cv2",&cv2,vector<string>{"rl1"});
    cp4.setPar("topo",vector<int>{cv2.oTopo()[0]*cv2.oTopo()[1]*cv2.oTopo()[2]});
    Relu mrl2(cp4);
    ml.addLayer("rl2",&mrl2,vector<string>{"cv2"});
// l3
    cp5.setPar("topo",vector<int>{cv2.oTopo()[0],cv2.oTopo()[1],cv2.oTopo()[2],48,2,2});
    cp5.setString("{stride=2;pad=0}");
    Convolution cv3(cp5);
    ml.addLayer("cv3",&cv3,vector<string>{"rl2"});

    cp3x3.setPar("topo", vector<int>{cv3.oTopo()[0]*cv3.oTopo()[1]*cv3.oTopo()[2]});
    BatchNorm bn3x3(cp3x3);
    ml.addLayer("bn3x3",&bn3x3,vector<string>{"cv3"});

    cp6.setPar("topo",vector<int>{cv3.oTopo()[0]*cv3.oTopo()[1]*cv3.oTopo()[2]});
    Relu mrl3(cp6);
    ml.addLayer("rl3",&mrl3,vector<string>{"bn3x3"});
// l4
    cp7.setPar("topo",vector<int>{cv3.oTopo()[0],cv3.oTopo()[1],cv3.oTopo()[2],48,3,3});
    cp7.setString("{stride=1;pad=1}");
    Convolution cv4(cp7);
    ml.addLayer("cv4",&cv4,vector<string>{"rl3"});
    cp8.setPar("topo",vector<int>{cv4.oTopo()[0]*cv4.oTopo()[1]*cv4.oTopo()[2]});
    Relu mrl4(cp8);
    ml.addLayer("rl4",&mrl4,vector<string>{"cv4"});
// l5
    cp9.setPar("topo",vector<int>{cv4.oTopo()[0],cv4.oTopo()[1],cv4.oTopo()[2],48,3,3});
    cp9.setString("{stride=1;pad=1}");
    Convolution cv5(cp9);
    ml.addLayer("cv5",&cv5,vector<string>{"rl4"});
    cp10.setPar("topo",vector<int>{cv5.oTopo()[0]*cv5.oTopo()[1]*cv5.oTopo()[2]});
    Relu mrl5(cp10);
    ml.addLayer("rl5",&mrl5,vector<string>{"cv5"});
// l6
    cp11.setPar("topo",vector<int>{cv5.oTopo()[0],cv5.oTopo()[1],cv5.oTopo()[2],48,2,2});
    cp11.setString("{stride=2;pad=0}");
    Convolution cv6(cp11);
    ml.addLayer("cv6",&cv6,vector<string>{"rl5"});
    cp12.setPar("topo",vector<int>{cv6.oTopo()[0]*cv6.oTopo()[1]*cv6.oTopo()[2]});
    Relu mrl6(cp12);
    ml.addLayer("rl6",&mrl6,vector<string>{"cv6"});
// l7
    cp13.setPar("topo",vector<int>{cv6.oTopo()[0]*cv6.oTopo()[1]*cv6.oTopo()[2],N4});
    Affine maf4(cp13);
    ml.addLayer("af4",&maf4,vector<string>{"rl6"});

    cp14.setPar("topo", vector<int>{N4});
    BatchNorm bn4(cp14);
    ml.addLayer("bn4",&bn4,vector<string>{"af4"});

    cp15.setPar("topo", vector<int>{N4});
    Relu mrl7(cp15);
    ml.addLayer("rl7",&mrl7,vector<string>{"bn4"});

    cp16.setPar("topo", vector<int>{N4});
    cp16.setPar("drop", dropR);
    Dropout dr4(cp16);
    ml.addLayer("dr4",&dr4,vector<string>{"rl7"});
// l8
    cp17.setPar("topo",vector<int>{N4,N5});
    Affine maf5(cp17);
    ml.addLayer("af5",&maf5,vector<string>{"dr4"});

    cp18.setPar("topo", vector<int>{N5});
    BatchNorm bn5(cp18);
    ml.addLayer("bn5",&bn5,vector<string>{"af5"});

    cp19.setPar("topo", vector<int>{N5});
    Relu mrl8(cp19);
    ml.addLayer("rl8",&mrl8,vector<string>{"bn5"});

    cp20.setPar("topo", vector<int>{N5});
    cp20.setPar("drop", dropR);
    Dropout dr5(cp20);
    ml.addLayer("dr5",&dr5,vector<string>{"rl8"});
// l9
    cp21.setPar("topo",vector<int>{N5,10});
    Affine maf6(cp21);
    ml.addLayer("af6",&maf6,vector<string>{"dr5"});

    Softmax msm1("{topo=[10]}");
    ml.addLayer("sm1",&msm1,vector<string>{"af6"});

    if (verbose) cout << "Checking multi-layer topology..." << endl;
    if (!ml.checkTopology(verbose)) {
        if (verbose) cout << "Topology-check for MultiLayer: ERROR." << endl;
    } else {
        if (verbose) cout << "Topology-check for MultiLayer: ok." << endl;
    }

    floatN cAcc=ml.train(X, y, Xv, yv, "Adam", cpo);
    floatN final_err;

    if (evalFinal) {
        final_err=ml.test(Xt, yt);
        cout << "Final error on test-set:" << final_err << ", accuracy:" << 1.0-final_err << endl;
        cAcc=1-final_err;
    }
    return cAcc;
}

std::vector<string> classes{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

void checkPrint(MatrixN& X, MatrixN& y, int id) {
    cout << "Class: " << y(id,0) << "=" << classes[y(id,0)] << endl;
    for (int i=0; i<3072; i++) {
        if (i%32==0) cout<<endl;
        if (i%(32*32)==0) cout<< "-------------" << endl;
        floatN p=X(id,i);
        if (p<-0.25) cout << " ";
        else if (p<0.0) cout << ".";
        else if (p<0.25) cout << "o";
        else cout << "#";
    }
}

int main(int argc, char *argv[]) {
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier gray(Color::FG_LIGHT_GRAY);
    Color::Modifier def(Color::FG_DEFAULT);

     if (argc!=2) {
         cout << "cifar10test <path-cifar10.h5-file>" << endl;
         exit(-1);
     }
     getcifar10Data(argv[1]);
     for (auto it : cpcifar10Data) {
         cout << it.first << " " <<  shape(*it.second) << endl;
     }
     for (auto it : cpcifar10Data4) {
         cout << it.first << " tensor-4" <<  endl;
     }


     cpInitCompute("Cifar10");

    MatrixN X=(*(cpcifar10Data["train-data"])).block(0,0,49000,3072);
    MatrixN y=(*(cpcifar10Data["train-labels"])).block(0,0,49000,1);
    MatrixN Xv=(*(cpcifar10Data["train-data"])).block(49000,0,1000,3072);
    MatrixN yv=(*(cpcifar10Data["train-labels"])).block(49000,0,1000,1);
    MatrixN Xt=*(cpcifar10Data["test-data"]);
    MatrixN yt=*(cpcifar10Data["test-labels"]);

    //checkPrint(X,y,10);
    //checkPrint(X,y,20);
    //checkPrint(X,y,30);

    //MultiLayer ml("{topo=[3072];name='multi1'}"); // unneeded. XXX remove the whole ifdef stuff, obsolete
    //createMultilayer(&ml);

    //Multilayer1

    CpParams cpo("{verbose=true;learning_rate=1e-2;lr_decay=1.0;momentum=0.9;decay_rate=0.98;epsion=1e-8}");
    cpo.setPar("epochs",(floatN)200.0);
    cpo.setPar("batch_size",50);
    cpo.setPar("regularization", (floatN)0.0); //0.0000001);
    floatN final_err;


    cpo.setPar("learning_rate", (floatN)2e-2); //2.2e-2);
    cpo.setPar("lr_decay", (floatN)1.0);
    cpo.setPar("regularization", (floatN)1e-5);

    bool autoOpt=false;

    floatN bReg, bLearn;
    if (autoOpt) {
        //vector<floatN> regi{1e-3,1e-4,1e-5,1e-6,1e-7}; -> 1e-5
        //vector<floatN> learni{5e-2,1e-2,5e-3,1e-3}; -> 1e-2
        vector<floatN> regi{1e-1,5e-2,1e-2,5e-3,1e-3,1e-4,1e-5,1e-6,1e-7}; // -> 2e-5 (2nd: 4e-5, 3rd 2e-5)
        vector<floatN> learni{1e-2,5e-3,1e-3}; // -> 1e-2 (2nd: 6e-3, 3rd 3e-3)
        cpo.setPar("epochs",(floatN)0.10);
        floatN cmAcc=0.0, cAcc;
        for (auto learn : learni) {
            cpo.setPar("learning_rate", learn);
            for (auto reg : regi) {
                cpo.setPar("regularization", reg);
                cAcc=evalMultilayer(cpo, X, y, Xv, yv, Xt, yt);
                if (cAcc > cmAcc) {
                    bReg=reg;
                    bLearn=learn;
                    cmAcc=cAcc;
                    cout << green << "Best: Acc:" << cmAcc << ", Reg:" << bReg << ", Learn:" << bLearn << def << endl;
                } else {
                    cout << red << "      Acc:" << cAcc << ", Reg:" << reg << ", Learn:" << learn << def << endl;
                }
            }
        }
        cout << endl << green << "Starting training with: Acc:" << cmAcc << ", Reg:" << bReg << ", Learn:" << bLearn << def << endl;
    } else {
        bLearn=1.e-2;
        bReg=1.e-3;
    }

    cpo.setPar("learning_rate", bLearn);
    cpo.setPar("regularization", bReg);
    cpo.setPar("epochs",(floatN)25.0);
    evalMultilayer(cpo, X, y, Xv, yv, Xt, yt, true, true);

    for (auto it : cpcifar10Data) {
         free(it.second);
         it.second=nullptr;
     }
     cpExitCompute();

     return 0;
 }

 /*    int id=15;
     cout << "Class: " << y(id,0) << "=" << classes[y(id,0)] << endl;
     for (int i=0; i<3072; i++) {
         if (i%32==0) cout<<endl;
         if (i%(32*32)==0) cout<< "-------------" << endl;
         floatN p=X(id,i);
         if (p<0.25) cout << " ";
         else if (p<0.5) cout << ".";
         else if (p<0.75) cout << "o";
         else cout << "#";
     }
     cout << endl << "=====================================================" << endl;

     id=20;
     cout << "Class: " << y(id,0) << "=" << classes[y(id,0)] << endl;
     for (int i=0; i<3072; i++) {
         if (i%32==0) cout<<endl;
         if (i%(32*32)==0) cout<< "-------------" << endl;
         floatN p=X(id,i);
         if (p<0.25) cout << " ";
         else if (p<0.5) cout << ".";
         else if (p<0.75) cout << "o";
         else cout << "#";
     }
     cout << endl << "=====================================================" << endl;

     id=25;
     cout << "Class: " << y(id,0) << "=" << classes[y(id,0)] << endl;
     for (int i=0; i<3072; i++) {
         if (i%32==0) cout<<endl;
         if (i%(32*32)==0) cout<< "-------------" << endl;
         floatN p=X(id,i);
         if (p<0.25) cout << " ";
         else if (p<0.5) cout << ".";
         else if (p<0.75) cout << "o";
         else cout << "#";
     }
     cout << endl << "=====================================================" << endl;
     for (int i=0; i<3072; i++) {
      cout << X(id,i) << ", ";
     }
     cout << endl;
*/
