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
            case H5T_INTEGER:
                cout << " int" << endl;
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

    TwoLayerNet tl({3072,1000,10});
    MatrixN X=(*(cpcifar10Data["train-data"])).block(0,0,49000,3072);
    MatrixN y=(*(cpcifar10Data["train-labels"])).block(0,0,49000,1);
    MatrixN Xv=(*(cpcifar10Data["train-data"])).block(49000,0,1000,3072);
    MatrixN yv=(*(cpcifar10Data["train-labels"])).block(49000,0,1000,1);
    MatrixN Xt=*(cpcifar10Data["test-data"]);
    MatrixN yt=*(cpcifar10Data["test-labels"]);

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
    cp_t_params<int> pi;
    cp_t_params<floatN> pf;
    pi["verbose"]=1;
    pi["epochs"]=40;
    pi["batch_size"]=400;
    pf["learning_rate"]=1e-2;
    pf["lr_decay"]=1.0;

    pf["momentum"]=0.9;

    pf["decay_rate"]=0.99;
    pf["epsilon"]=1e-8;

    pi["threads"]=8;
    tl.train(X, y, Xv, yv, "Adam", pi, pf);
    floatN final_err=tl.test(Xt, yt);
    cout << "Final error on test-set:" << final_err << endl;
    for (auto it : cpcifar10Data) {
         free(it.second);
         it.second=nullptr;
     }
     return 0;
 }
