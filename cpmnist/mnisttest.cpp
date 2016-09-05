#include <iostream>
#include <string>
#include <H5Cpp.h>
#include <H5File.h>
#include <H5DataSet.h>

#include "cp-neural.h"

typedef struct {
    int index;               // index of the current object
} cp_mnist_iter_info;

herr_t cp_mnist_get_all_groups(hid_t loc_id, const char *name, void *opdata);

H5::H5File *cp_Mnist_pfile;
map<string, MatrixN *> cpMnistData;

herr_t cp_mnist_get_all_groups(hid_t loc_id, const char *name, void *opdata)
{
    cp_mnist_iter_info *info=(cp_mnist_iter_info *)opdata;

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
        // cout << " is a dataset" << endl;
        H5::DataSet dataset = cp_Mnist_pfile->openDataSet(name);
        /*
         * Get filespace for rank and dimension
         */
        H5::DataSpace filespace = dataset.getSpace();
        /*
         * Get number of dimensions in the file dataspace
         */
        int rank = filespace.getSimpleExtentNdims();
        // cout << "rank: " << rank << endl;
        /*
         * Get and print the dimension sizes of the file dataspace
         */
        hsize_t dims[2];    // dataset dimensions
        rank = filespace.getSimpleExtentDims( dims );
        // cout << "dataset rank = " << rank << ", dimensions; ";
        // for (int j=0; j<rank; j++) {
        //    cout << (unsigned long)(dims[j]);
        //    if (j<rank-1) cout << " x ";
        //}
        //cout << endl;
        if (rank==1) dims[1]=1;
        cpMnistData[name] = new MatrixN(dims[0],dims[1]);
        MatrixN *pM = cpMnistData[name];
        /*
         * Define the memory space to read dataset.
         */
        H5::DataSpace mspace1(rank, dims);
        /*
         * Read dataset back and display.
         */
         auto dataClass = dataset.getTypeClass();

         if(dataClass == H5T_FLOAT)
         {
             auto floatType = dataset.getFloatType();

             size_t byteSize = floatType.getSize();

             if(byteSize == 4)
             {
                  float *pf = (float *)malloc(sizeof(float) * dims[0] * dims[1]);
                  if (pf) {
                      dataset.read(pf, H5::PredType::NATIVE_FLOAT, mspace1, filespace );
                      for (int y=0; y<dims[0]; y++) {
                          for (int x=0; x<dims[1]; x++) {
                              (*pM)(y,x)=(floatN)pf[y*dims[1]+x];
                          }
                      }
                      free(pf);
                  }
             }
             else if(byteSize == 8)
             {
                 double *pd = (double *)malloc(sizeof(double) * dims[0] * dims[1]);
                 if (pd) {
                     dataset.read(pd, H5::PredType::NATIVE_DOUBLE, mspace1, filespace );
                     for (int y=0; y<dims[0]; y++) {
                         for (int x=0; x<dims[1]; x++) {
                             (*pM)(y,x)=(floatN)pd[y*dims[1]+x];
                         }
                     }
                     free(pd);
                 }
             }
         } else if (dataClass == H5T_INTEGER) {
             int *pi = (int *)malloc(sizeof(int) * dims[0] * dims[1]);
             if (pi) {
                 dataset.read(pi, H5::PredType::NATIVE_INT, mspace1, filespace );
                 for (int y=0; y<dims[0]; y++) {
                     for (int x=0; x<dims[1]; x++) {
                         (*pM)(y,x)=(floatN)pi[y*dims[1]+x];
                     }
                 }
                 free(pi);
             }
         }
    }
    (info->index)++;
    return 0;
 }

bool  getMnistData(string filepath) {
    if (cpMnistData.size() > 0) {
        cout << "cpMnistData contains already elements, not reloading." << endl;
        return true;
    }
    H5::H5File fmn((H5std_string)filepath, H5F_ACC_RDONLY);
    cp_Mnist_pfile=&fmn;
    int nr=fmn.getNumObjs();
    //cout << nr << endl;
    cp_mnist_iter_info info;
    info.index=0;
    cout << "Reading: ";
    fmn.iterateElems("/", NULL, cp_mnist_get_all_groups, &info);
    cout << endl;
    return true;
}

 int main(int argc, char *argv[]) {
     if (argc!=2) {
         cout << "mnisttest <path-mnist.h5-file>" << endl;
         exit(-1);
     }
     getMnistData(argv[1]);
     for (auto it : cpMnistData) {
         cout << it.first << " " << shape(*(it.second)) << endl;
     }

     /*
     vector<int> ins{0,4,16,25,108, 256,777};
     for (auto in : ins) {
         cout << "-------------------------" << endl;
         cout << "Index: " << in << endl;
         for (int cy=0; cy<28; cy++) {
             for (int cx=0; cx<28; cx++) {
                 floatN pt=(*cpMnistData["x_test"])(in, cy*28+cx);
                 if (pt<0.5) cout << " ";
                 else cout << "*";
             }
             cout << endl;
         }
         cout << (*cpMnistData["t_test"])(in,0) << endl;
     }
     */
    TwoLayerNet tl({784,1024,10});
    MatrixN X=*(cpMnistData["x_train"]);
    MatrixN y=*(cpMnistData["t_train"]);
    MatrixN Xv=*(cpMnistData["x_valid"]);
    MatrixN yv=*(cpMnistData["t_valid"]);
    MatrixN Xt=*(cpMnistData["x_test"]);
    MatrixN yt=*(cpMnistData["t_test"]);
    cp_t_params<int> pi;
    cp_t_params<floatN> pf;
    pi["verbose"]=1;
    pi["epochs"]=2;
    pi["batch_size"]=400;
    pf["learning_rate"]=1e-3;
    pf["lr_decay"]=1.0;

    pf["momentum"]=0.9;

    pf["decay_rate"]=0.98;
    pf["epsilon"]=1e-8;

    pi["threads"]=4;
    tl.train(X, y, Xv, yv, "Adam", pi, pf);
    floatN final_err=tl.test(Xt, yt);
    cout << "Final error on test-set:" << final_err << endl;
    for (auto it : cpMnistData) {
         free(it.second);
     }
     return 0;
 }
