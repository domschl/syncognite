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
                    dataset.read(pi4, H5::PredType::NATIVE_INT, mspace1, filespace );
                    for (int z=0; z<dims[0]; z++) {
                        for (int y=0; y<dims[1]; y++) {
                            for (int x=0; x<dims[2]; x++) {
                                for (int w=0; w<dims[3]; w++) {
                                    (*(cpcifar10Data4[name]))(z,y,x,w)=(floatN)pi4[z*dims[3]*dims[2]*dims[1] + y*dims[3]*dims[2]+x*dims[3] + w];
                                    (*(cpcifar10Data[name]))(z,y*dims[3]*dims[2]+x*dims[3]+w)=(floatN)pi4[z*dims[3]*dims[2]*dims[1] + y*dims[3]*dims[2]+x*dims[3] + w] / 256.0;
                                }
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


std::vector<string> classes{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
int main(int argc, char *argv[]) {
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

     cpInitCompute();

    MatrixN X=(*(cpcifar10Data["train-data"])).block(0,0,49000,3072);
    MatrixN y=(*(cpcifar10Data["train-labels"])).block(0,0,49000,1);
    MatrixN Xv=(*(cpcifar10Data["train-data"])).block(49000,0,1000,3072);
    MatrixN yv=(*(cpcifar10Data["train-labels"])).block(49000,0,1000,1);
    MatrixN Xt=*(cpcifar10Data["test-data"]);
    MatrixN yt=*(cpcifar10Data["test-labels"]);

    //#define USE_2LN 1
    #ifdef USE_2LN
    TwoLayerNet tl(CpParams("{topo=[3072,1000,10]}"));
    #else
    //Multilayer1
    int N0=3072,N1=1200,N2=1000,N3=700, N4=500, N5=200;
    MultiLayer ml("{topo=[3072];name='multi1'}");
    cout << "LayerName for ml: " << ml.layerName << endl;
    CpParams cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9,cp10,cp11,cp12,cp13,cp14,cp15,cp16,cp17,cp18,cp19,cp20,cp21;
    floatN dropR=0.5;
// l1
    cp1.setPar("topo",vector<int>{N0,N1});
    Affine maf1(cp1);
    ml.addLayer("af1",&maf1,vector<string>{"input"});

    cp2.setPar("topo", vector<int>{N1});
    BatchNorm bn1(cp2);
    ml.addLayer("bn1",&bn1,vector<string>{"af1"});

    cp3.setPar("topo", vector<int>{N1});
    Relu mrl1(cp3);
    ml.addLayer("rl1",&mrl1,vector<string>{"bn1"});

    cp4.setPar("topo", vector<int>{N1});
    cp4.setPar("drop", dropR);
    Dropout dr1(cp4);
    ml.addLayer("dr1",&dr1,vector<string>{"rl1"});
// l2
    cp5.setPar("topo",vector<int>{N1,N2});
    Affine maf2(cp5);
    ml.addLayer("af2",&maf2,vector<string>{"dr1"});

    cp6.setPar("topo", vector<int>{N2});
    BatchNorm bn2(cp6);
    ml.addLayer("bn2",&bn2,vector<string>{"af2"});

    cp7.setPar("topo", vector<int>{N2});
    Relu mrl2(cp7);
    ml.addLayer("rl2",&mrl2,vector<string>{"bn2"});

    cp8.setPar("topo", vector<int>{N2});
    cp8.setPar("drop", dropR);
    Dropout dr2(cp8);
    ml.addLayer("dr2",&dr2,vector<string>{"rl2"});
// l3
    cp9.setPar("topo",vector<int>{N2,N3});
    Affine maf3(cp9);
    ml.addLayer("af3",&maf3,vector<string>{"dr2"});

    cp10.setPar("topo", vector<int>{N3});
    BatchNorm bn3(cp10);
    ml.addLayer("bn3",&bn3,vector<string>{"af3"});

    cp11.setPar("topo", vector<int>{N3});
    Relu mrl3(cp11);
    ml.addLayer("rl3",&mrl3,vector<string>{"bn3"});

    cp12.setPar("topo", vector<int>{N3});
    cp12.setPar("drop", dropR);
    Dropout dr3(cp12);
    ml.addLayer("dr3",&dr3,vector<string>{"rl3"});
// l4
    cp13.setPar("topo",vector<int>{N3,N4});
    Affine maf4(cp13);
    ml.addLayer("af4",&maf4,vector<string>{"dr3"});

    cp14.setPar("topo", vector<int>{N4});
    BatchNorm bn4(cp14);
    ml.addLayer("bn4",&bn4,vector<string>{"af4"});

    cp15.setPar("topo", vector<int>{N4});
    Relu mrl4(cp15);
    ml.addLayer("rl4",&mrl4,vector<string>{"bn4"});

    cp16.setPar("topo", vector<int>{N4});
    cp16.setPar("drop", dropR);
    Dropout dr4(cp16);
    ml.addLayer("dr4",&dr4,vector<string>{"rl4"});
// l5
    cp17.setPar("topo",vector<int>{N4,N5});
    Affine maf5(cp17);
    ml.addLayer("af5",&maf5,vector<string>{"dr4"});

    cp18.setPar("topo", vector<int>{N5});
    BatchNorm bn5(cp18);
    ml.addLayer("bn5",&bn5,vector<string>{"af5"});

    cp19.setPar("topo", vector<int>{N5});
    Relu mrl5(cp19);
    ml.addLayer("rl5",&mrl5,vector<string>{"bn5"});

    cp20.setPar("topo", vector<int>{N5});
    cp20.setPar("drop", dropR);
    Dropout dr5(cp20);
    ml.addLayer("dr5",&dr5,vector<string>{"rl5"});
// l6
    cp21.setPar("topo",vector<int>{N5,10});
    Affine maf6(cp21);
    ml.addLayer("af6",&maf6,vector<string>{"dr5"});

    Softmax msm1("{topo=[10]}");
    ml.addLayer("sm1",&msm1,vector<string>{"af6"});
    if (!ml.checkTopology()) {
        cout << "Topology-check for MultiLayer: ERROR." << endl;
    } else {
        cout << "Topology-check for MultiLayer: ok." << endl;
    }
    #endif

    CpParams cpo("{verbose=true;learning_rate=1e-2;lr_decay=1.0;momentum=0.9;decay_rate=0.98;epsion=1e-8}");
    cpo.setPar("epochs",200);
    cpo.setPar("batch_size",200);
    cpo.setPar("regularization", (floatN)0.0); //0.0000001);
    floatN final_err;


    #ifdef USE_2LN
    cpo.setPar("learning_rate", (floatN)1e-2);
    tl.train(X, y, Xv, yv, "Adam", cpo);
    final_err=tl.test(Xt, yt);
    #else
    cpo.setPar("learning_rate", (floatN)1e-3); //2.2e-2);
    cpo.setPar("regularization", (floatN)1e-4);
    ml.train(X, y, Xv, yv, "Adam", cpo);
    final_err=ml.test(Xt, yt);
    #endif
    cout << "Final error on test-set:" << final_err << ", accuracy:" << 1.0-final_err << endl;
    for (auto it : cpcifar10Data) {
         free(it.second);
         it.second=nullptr;
     }
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
