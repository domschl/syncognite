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

     cpInitCompute("Mnist");
     registerLayers();

     MatrixN X=*(cpMnistData["x_train"]);
     MatrixN y=*(cpMnistData["t_train"]);
     MatrixN Xv=*(cpMnistData["x_valid"]);
     MatrixN yv=*(cpMnistData["t_valid"]);
     MatrixN Xt=*(cpMnistData["x_test"]);
     MatrixN yt=*(cpMnistData["t_test"]);

     LayerBlock ms("name='DomsNet'");
     ms.addLayer("Convolution", "cv1", "{topo=[1,28,28];kernel=[16,3,3];stride=1;pad=1}",vector<string>{"input"});
     ms.addLayer("BatchNorm","sb1","",vector<string>{"cv1"});
     ms.addLayer("Relu","rl1","",vector<string>{"sb1"});
     ms.addLayer("Dropout","doc1","{drop=0.5}",vector<string>{"rl1"});
     ms.addLayer("Convolution", "cv2", "{topo=[0,0,0];kernel=[32,3,3];stride=1;pad=1}",vector<string>{"doc1"});
     ms.addLayer("Relu","rl2","",vector<string>{"cv2"});
     ms.addLayer("Convolution", "cv3", "{topo=[0,0,0];kernel=[64,3,3];stride=2;pad=1}",vector<string>{"rl2"});
     ms.addLayer("BatchNorm","sb2","",vector<string>{"cv3"});
     ms.addLayer("Relu","rl3","",vector<string>{"sb2"});
     ms.addLayer("Dropout","doc2","{drop=0.5}",vector<string>{"rl3"});
     ms.addLayer("Convolution", "cv4", "{topo=[0,0,0];kernel=[64,3,3];stride=1;pad=1}",vector<string>{"doc2"});
     ms.addLayer("Relu","rl4","",vector<string>{"cv4"});
     ms.addLayer("Convolution", "cv5", "{topo=[0,0,0];kernel=[128,3,3];stride=2;pad=1}",vector<string>{"rl4"});
     ms.addLayer("BatchNorm","sb3","",vector<string>{"cv5"});
     ms.addLayer("Relu","rl5","",vector<string>{"sb3"});
     ms.addLayer("Dropout","doc3","{drop=0.5}",vector<string>{"rl5"});
     //ms.addLayer("Convolution", "cv6", "{topo=[0,0,0];kernel=[64,3,3];stride=2;pad=1}",vector<string>{"rl5"});
     //ms.addLayer("Relu","rl6","",vector<string>{"cv6"});
     //ms.addLayer("Convolution", "cv7", "{topo=[0,0,0];kernel=[64,3,3];stride=1;pad=1}",vector<string>{"rl6"});
     //ms.addLayer("Relu","rl7","",vector<string>{"cv7"});

     ms.addLayer("Affine","af1","{topo=[0];hidden=1024}",vector<string>{"doc3"});
     ms.addLayer("BatchNorm","bn1","",vector<string>{"af1"});
     ms.addLayer("Relu","rla1","",vector<string>{"bn1"});
     ms.addLayer("Dropout","do1","{drop=0.7}",vector<string>{"rla1"});
     ms.addLayer("Affine","af2","{topo=[0];hidden=512}",vector<string>{"do1"});
     ms.addLayer("BatchNorm","bn2","",vector<string>{"af2"});
     ms.addLayer("Relu","rla2","",vector<string>{"bn2"});
     ms.addLayer("Dropout","do2","{drop=0.7}",{"rla2"});
     ms.addLayer("Affine","af3","{topo=[0];hidden=10}",vector<string>{"do2"});
     ms.addLayer("Softmax","sm1","",vector<string>{"af3"});

     bool verbose=true;
     if (verbose) cout << "Checking multi-layer topology..." << endl;
     if (!ms.checkTopology(verbose)) {
         cout << "Topology-check for MultiLayer: ERROR." << endl;
         exit(-1);
     } else {
         if (verbose) cout << "Topology-check for MultiLayer: ok." << endl;
     }

     CpParams cpo("{verbose=true;lr_decay=0.95;epsilon=1e-8}");
     cpo.setPar("epochs",(floatN)40.0);
     cpo.setPar("batch_size",25);
     cpo.setPar("learning_rate", (floatN)1e-4);
     cpo.setPar("regularization", (floatN)1e-7);

     ms.train(X, y, Xv, yv, "Adam", cpo);

     floatN train_err, val_err, test_err;
     train_err=ms.test(X, y, cpo.getPar("batch_size", 50));
     val_err=ms.test(Xv, yv, cpo.getPar("batch_size", 50));
     test_err=ms.test(Xt, yt, cpo.getPar("batch_size", 50));

     cout << "Final results on MNIST after " << cpo.getPar("epochs",0) << " epochs:" << endl;
     cout << "      Train-error: " << train_err << endl;
     cout << " Validation-error: " << val_err << endl;
     cout << "       Test-error: " << test_err << endl;

     for (auto it : cpMnistData) {
          free(it.second);
          it.second=nullptr;
      }
     cpExitCompute();
 }

/*
//#define USE_2LN 1
#ifdef USE_2LN
        TwoLayerNet tl(CpParams("{topo=[784,1024,10]}"));
#else
        //Multilayer1
        int N0=784,N1=1024,N2=512,N3=128;
        MultiLayer ml("{topo=[784];name='multi1'}");
        cout << "LayerName for ml: " << ml.layerName << endl;
        CpParams cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9,cp10,cp11,cp12,cp13;
//l1
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
        cp4.setPar("drop", (floatN)0.7);
        Dropout dr1(cp4);
        ml.addLayer("dr1",&dr1,vector<string>{"rl1"});
//l2
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
        cp8.setPar("drop", (floatN)0.8);
        Dropout dr2(cp8);
        ml.addLayer("dr2",&dr2,vector<string>{"rl2"});
//l3
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
        cp12.setPar("drop", (floatN)0.9);
        Dropout dr3(cp12);
        ml.addLayer("dr3",&dr3,vector<string>{"rl3"});
//l4
        cp13.setPar("topo",vector<int>{N3,10});
        Affine maf4(cp13);
        ml.addLayer("af4",&maf4,vector<string>{"dr3"});

        Softmax msm1("{topo=[10]}");
        ml.addLayer("sm1",&msm1,vector<string>{"af4"});
        if (!ml.checkTopology()) {
            cout << "Topology-check for MultiLayer: ERROR." << endl;
        } else {
            cout << "Topology-check for MultiLayer: ok." << endl;
        }
#endif
    MatrixN X=*(cpMnistData["x_train"]);
    MatrixN y=*(cpMnistData["t_train"]);
    MatrixN Xv=*(cpMnistData["x_valid"]);
    MatrixN yv=*(cpMnistData["t_valid"]);
    MatrixN Xt=*(cpMnistData["x_test"]);
    MatrixN yt=*(cpMnistData["t_test"]);

    CpParams cpo("{verbose=true;learning_rate=1e-2;lr_decay=1.0;momentum=0.9;decay_rate=0.98;epsion=1e-8}");
    cpo.setPar("epochs",(floatN)200.0);
    cpo.setPar("batch_size",50);
    cpo.setPar("regularization", (floatN)0.0); //0.00001);
    floatN final_err;

#ifdef USE_2LN
    cpo.setPar("learning_rate", (floatN)1e-3);
    cpo.setPar("regularization", 0.0);
    tl.train(X, y, Xv, yv, "Adam", cpo);
    final_err=tl.test(Xt, yt,50);
#else
    cpo.setPar("learning_rate", (floatN)5e-2);
    cpo.setPar("regularization", (floatN)1e-6);
    ml.train(X, y, Xv, yv, "Adam", cpo);
    final_err=ml.test(Xt, yt, cpo.getPar("batch_size", 50));
#endif
    cout << "Final error on test-set:" << final_err << endl;
    for (auto it : cpMnistData) {
         free(it.second);
         it.second=nullptr;
     }
     cpExitCompute();
     return 0;
 }*/

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
