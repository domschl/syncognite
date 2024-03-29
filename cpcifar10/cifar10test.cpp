#include <iostream>
#include <string>
#include <H5Cpp.h>
#include <H5File.h>
#include <H5DataSet.h>

#include "cp-neural.h"
//#include <unsupported/Eigen/CXX11/Tensor>

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
    cerr << name << " ";

    // you can use this call to select just the groups
    // H5G_LINK    0  Object is a symbolic link.
    // H5G_GROUP   1  Object is a group.
    // H5G_DATASET 2  Object is a dataset.
    // H5G_TYPE    3  Object is a named datatype.
    int obj_type = H5Gget_objtype_by_idx(loc_id, info->index);
    //if(obj_type == H5G_GROUP)
	   //cerr << "        is a group" << endl;
    if (obj_type == H5G_DATASET) {
        H5::DataSet dataset = cp_cifar10_pfile->openDataSet(name);
        H5::DataSpace filespace = dataset.getSpace();
        int rank; // = filespace.getSimpleExtentNdims();
        // cerr << "rank: " << rank << endl;
        hsize_t dims[10];    // dataset dimensions
        rank = filespace.getSimpleExtentDims( dims );
        cerr << "dataset rank = " << rank << ", dimensions; ";
        for (int j=0; j<rank; j++) {
            cerr << (unsigned long)(dims[j]);
            if (j<rank-1) cerr << " x ";
        }
        H5::DataSpace mspace1(rank, dims);
        auto dataClass = dataset.getTypeClass();
        switch (dataClass) {
            case H5T_FLOAT:
                if (dataset.getFloatType().getSize()==4) {
                    cerr << " float" << endl;
                } else if (dataset.getFloatType().getSize()==8) {
                    cerr << " double" << endl;
                }
                break;
            case H5T_INTEGER:
                cerr << " int" << endl;
                break;
            default:
                cerr << "dataClass not implemented: " << dataClass << endl;
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
                                        cerr << "[" << pi4[z*dims[3]*dims[2]*dims[1] + y*dims[3]*dims[2]+x*dims[3] + w] << " -> " << (*(cpcifar10Data[name]))(z,y*dims[3]*dims[2]+x*dims[3]+w) << "]  " ;
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
                cerr << "NOT YET IMPLEMENTED RANK: " << rank << endl;
                exit(-1);
        }
    }

    (info->index)++;

return 0;
 }

bool  getcifar10Data(string filepath) {
    if (cpcifar10Data.size() > 0) {
        cerr << "cpcifar10Data contains already elements, not reloading." << endl;
        return true;
    }
    H5::H5File fmn((H5std_string)filepath, H5F_ACC_RDONLY);
    cp_cifar10_pfile=&fmn;
    //int nr=fmn.getNumObjs();
    //cerr << nr << endl;
    cp_cifar10_iter_info info;
    info.index=0;
    cerr << "Reading: ";
    fmn.iterateElems("/", NULL, cp_cifar10_get_all_groups, &info);
    cerr << endl;
    return true;
}


floatN evalMultilayer(json& jo, MatrixN& X, MatrixN& y, MatrixN& Xv, MatrixN& yv, MatrixN& Xt, MatrixN& yt, 
                      Optimizer *pOpt, t_cppl *pOptimizerState, Loss *pLoss, 
                      bool evalFinal=false, bool verbose=false, int mode=0) {
    LayerBlockOldStyle lb(R"({"name":"DomsNet","bench":false,"init":"orthonormal","initfactor":0.1})"_json);
    if (mode==0) {
        lb.addLayer("Convolution", "cv1", R"({"inputShape":[3,32,32],"kernel":[64,5,5],"stride":1,"pad":2})",{"input"});
        lb.addLayer("BatchNorm","sb1","{}",{"cv1"});
        lb.addLayer("Relu","rl1","{}",{"sb1"});
        lb.addLayer("Dropout","doc1",R"({"drop":0.5})",{"rl1"});
        lb.addLayer("Convolution", "cv2", R"({"kernel":[64,3,3],"stride":1,"pad":1})",{"doc1"});
        lb.addLayer("Relu","rl2","{}",{"cv2"});
        lb.addLayer("Convolution", "cv3", R"({"kernel":[128,3,3],"stride":2,"pad":1})",{"rl2"});
        lb.addLayer("BatchNorm","sb2","{}",{"cv3"});
        lb.addLayer("Relu","rl3","{}",{"sb2"});
        lb.addLayer("Dropout","doc2",R"({"drop":0.8})",{"rl3"});
        lb.addLayer("Convolution", "cv4", R"({"kernel":[128,3,3],"stride":1,"pad":1})",{"doc2"});
        lb.addLayer("Relu","rl4","{}",{"cv4"});
        lb.addLayer("Convolution", "cv5", R"({"kernel":[256,3,3],"stride":2,"pad":1})",{"rl4"});
        lb.addLayer("BatchNorm","sb3","{}",{"cv5"});
        lb.addLayer("Relu","rl5","{}",{"sb3"});
        lb.addLayer("Dropout","doc3",R"({"drop":0.6})",{"rl5"});
        lb.addLayer("Convolution", "cv6", R"({"kernel":[256,3,3],"stride":1,"pad":1})",{"doc3"});
        lb.addLayer("Relu","rl6","{}",{"cv6"});
        lb.addLayer("Dropout","doc4",R"({"drop":0.6})",{"rl6"});
        lb.addLayer("Convolution", "cv7", R"({"kernel":[512,3,3],"stride":2,"pad":1})",{"doc4"});
        lb.addLayer("Relu","rl7","{}",{"cv7"});
        lb.addLayer("Dropout","doc5",R"({"drop":0.6})",{"rl7"});
        lb.addLayer("Convolution", "cv8", R"({"kernel":[512,3,3],"stride":1,"pad":1})",{"doc5"});
        lb.addLayer("Relu","rl8","{}",{"cv8"});

        lb.addLayer("Affine","af1",R"({"hidden":1024})",{"rl8"});
        lb.addLayer("BatchNorm","bn1","{}",{"af1"});
        lb.addLayer("Relu","rla1","{}",{"bn1"});
        lb.addLayer("Dropout","do1",R"({"drop":0.7})",{"rla1"});
        lb.addLayer("Affine","af2",R"({"hidden":512})",{"do1"});
        lb.addLayer("BatchNorm","bn2","{}",{"af2"});
        lb.addLayer("Relu","rla2","{}",{"bn2"});
        lb.addLayer("Dropout","do2",R"({"drop":0.7})",{"rla2"});
        lb.addLayer("Affine","af3",R"({"hidden":10})",{"do2"});
        lb.addLayer("Softmax","sm1","{}",{"af3"});
    } else if (mode==1) {
        lb.addLayer("Convolution", "cv1", R"({"inputShape":[3,32,32],"kernel":[64,5,5],"stride":1,"pad":2})",{"input"});
        lb.addLayer("Nonlinearity","nl1",R"({"type":"selu"})",{"cv1"});
        lb.addLayer("Dropout","doc1",R"({"drop":0.5})",{"nl1"});
        lb.addLayer("Convolution", "cv2", R"({"kernel":[64,3,3],"stride":1,"pad":1})",{"doc1"});
        lb.addLayer("Nonlinearity","nl2",R"({"type":"selu"})",{"cv2"});
        lb.addLayer("Convolution", "cv3", R"({"kernel":[128,3,3],"stride":2,"pad":1})",{"nl2"});
        lb.addLayer("Nonlinearity","nl3",R"({"type":"selu"})",{"cv3"});
        lb.addLayer("Dropout","doc2",R"({"drop":0.6})",{"nl3"});
        lb.addLayer("Convolution", "cv4", R"({"kernel":[128,3,3],"stride":1,"pad":1})",{"doc2"});
        lb.addLayer("Nonlinearity","nl4",R"({"type":"selu"})",{"cv4"});
        lb.addLayer("Convolution", "cv5", R"({"kernel":[256,3,3],"stride":2,"pad":1})",{"nl4"});
        lb.addLayer("Nonlinearity","nl5",R"({"type":"selu"})",{"cv5"});
        lb.addLayer("Dropout","doc3",R"({"drop":0.6})",{"nl5"});
        lb.addLayer("Convolution", "cv6", R"({"kernel":[256,3,3],"stride":1,"pad":1})",{"doc3"});
        lb.addLayer("Nonlinearity","nl6",R"({"type":"selu"})",{"cv6"});
        lb.addLayer("Dropout","doc4",R"({"drop":0.6})",{"nl6"});
        lb.addLayer("Convolution", "cv7", R"({"kernel":[512,3,3],"stride":2,"pad":1})",{"doc4"});
        lb.addLayer("Nonlinearity","nl7",R"({"type":"selu"})",{"cv7"});
        lb.addLayer("Dropout","doc5",R"({"drop":0.6})",{"nl7"});
        lb.addLayer("Convolution", "cv8", R"({"kernel":[512,3,3],"stride":1,"pad":1})",{"doc5"});
        lb.addLayer("Nonlinearity","nl8",R"({"type":"selu"})",{"cv8"});

        lb.addLayer("Affine","af1",R"({"hidden":1024})",{"nl8"});
        lb.addLayer("Nonlinearity","nla1",R"({"type":"selu"})",{"af1"});
        lb.addLayer("Dropout","do1",R"({"drop":0.7})",{"nla1"});
        lb.addLayer("Affine","af2",R"({"hidden":512})",{"do1"});
        lb.addLayer("Nonlinearity","nla2",R"({"type":"selu"})",{"af2"});
        lb.addLayer("Dropout","do2",R"({"drop":0.7})",{"nla2"});
        lb.addLayer("Affine","af3",R"({"hidden":10})",{"do2"});
        lb.addLayer("Softmax","sm1","{}",{"af3"});
    } else if (mode==2) {
        lb.addLayer("Convolution", "cv1", R"({"inputShape":[3,32,32],"kernel":[64,5,5],"stride":1,"pad":2})",{"input"});
        lb.addLayer("Nonlinearity","nl1s",R"({"type":"resilu"})",{"cv1"});
        lb.addLayer("BatchNorm","sb1","{}",{"nl1s"});
        lb.addLayer("Convolution", "cv2", R"({"kernel":[64,3,3],"stride":1,"pad":1})",{"sb1"});
        lb.addLayer("Nonlinearity","nl2",R"({"type":"resilu"})",{"cv2"});
        lb.addLayer("Convolution", "cv3", R"({"kernel":[128,3,3],"stride":2,"pad":1})",{"nl2"});
        lb.addLayer("Nonlinearity","nl3",R"({"type":"resilu"})",{"cv3"});
        lb.addLayer("BatchNorm","sb2","{}",{"nl3"});
        lb.addLayer("Convolution", "cv4", R"({"kernel":[128,3,3],"stride":1,"pad":1})",{"sb2"});
        lb.addLayer("Nonlinearity","nl4",R"({"type":"resilu"})",{"cv4"});
        lb.addLayer("Convolution", "cv5", R"({"kernel":[256,3,3],"stride":2,"pad":1})",{"nl4"});
        lb.addLayer("Nonlinearity","nl5",R"({"type":"resilu"})",{"cv5"});
        lb.addLayer("BatchNorm","sb21","{}",{"nl5"});
        
        lb.addLayer("Convolution", "cv6", R"({"kernel":[256,3,3],"stride":1,"pad":1})",{"sb21"});
        lb.addLayer("Nonlinearity","nl6",R"({"type":"resilu"})",{"cv6"});
        lb.addLayer("Convolution", "cv7", R"({"kernel":[512,3,3],"stride":2,"pad":1})",{"nl6"});
        lb.addLayer("Nonlinearity","nl7",R"({"type":"resilu"})",{"cv7"});
        lb.addLayer("BatchNorm","sb22","{}",{"nl7"});
        lb.addLayer("Convolution", "cv8", R"({"kernel":[512,3,3],"stride":1,"pad":1})",{"sb22"});
        lb.addLayer("Nonlinearity","nl8",R"({"type":"resilu"})",{"cv8"});
        
        lb.addLayer("Affine","af1",R"({"hidden":1024})",{"nl8"});
        lb.addLayer("Nonlinearity","nla1",R"({"type":"resilu"})",{"af1"});
        lb.addLayer("BatchNorm","bn1","{}",{"nla1"});
        lb.addLayer("Affine","af2",R"({"hidden":512})",{"bn1"});
        lb.addLayer("Nonlinearity","nla2",R"({"type":"resilu"})",{"af2"});
        lb.addLayer("Affine","af3",R"({"hidden":10})",{"nla2"});
        lb.addLayer("Softmax","sm1","{}",{"af3"});
    } else {
        cerr << "Bad mode for evalMultilayer! " << mode << endl;
        exit(-1);
    }

    if (verbose) cerr << "Checking LayerBlock topology..." << endl;
    if (!lb.checkTopology(verbose)) {
        if (verbose) cerr << "Topology-check for LayerBlock: ERROR." << endl;
    } else {
        if (verbose) cerr << "Topology-check for LayerBLock: ok." << endl;
    }

    floatN cAcc=lb.train(X, y, Xv, yv, pOpt, pOptimizerState, pLoss, jo);
    floatN train_err, val_err, test_err;
    if (evalFinal) {
        train_err=lb.test(X, y, jo.value("batch_size", 50));
        val_err=lb.test(Xv, yv, jo.value("batch_size", 50));
        test_err=lb.test(Xt, yt, jo.value("batch_size", 50));

        cerr << "Final results on CIFAR10 after " << jo.value("epochs",(floatN)0.0) << " epochs:" << endl;
        cerr << "      Train-error: " << train_err << " train-acc: " << 1.0-train_err << endl;
        cerr << " Validation-error: " << val_err <<   "   val-acc: " << 1.0-val_err << endl;
        cerr << "       Test-error: " << test_err <<  "  test-acc: " << 1.0-test_err << endl;
    }
    return cAcc;
}

std::vector<string> classes{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

void checkPrint(MatrixN& X, MatrixN& y, int id) {
    cerr << "Class: " << y(id,0) << "=" << classes[y(id,0)] << endl;
    for (int i=0; i<3072; i++) {
        if (i%32==0) cerr<<endl;
        if (i%(32*32)==0) cerr<< "-------------" << endl;
        floatN p=X(id,i);
        if (p<-0.25) cerr << " ";
        else if (p<0.0) cerr << ".";
        else if (p<0.25) cerr << "o";
        else cerr << "#";
    }
}

int main(int argc, char *argv[]) {
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier gray(Color::FG_LIGHT_GRAY);
    Color::Modifier def(Color::FG_DEFAULT);

    if (argc<2) {
        cerr << "cifar10test <path-cifar10.h5-file> [nonlinmode]" << endl;
        cerr << "nonlinmode=0: relu+batchnorm, 1: only SELU (https://arxiv.org/abs/1706.02515), 2: resilu" << endl;
        exit(-1);
    }
    getcifar10Data(argv[1]);
    int mode=0;
    if (argc>2) mode=std::stoi(argv[2]);
    if (mode<0 || mode>2) {
        cerr << "bad nonlin mode %d, " << mode << endl;
        exit(-1);
    }
    for (auto it : cpcifar10Data) {
        cerr << it.first << " " <<  shape(*it.second) << endl;
    }
    for (auto it : cpcifar10Data4) {
        cerr << it.first << " tensor-4" <<  endl;
    }

    cpInitCompute("Cifar10");
    registerLayers();

    MatrixN X=(*(cpcifar10Data["train-data"])).block(0,0,49000,3072);
    MatrixN y=(*(cpcifar10Data["train-labels"])).block(0,0,49000,1);
    MatrixN Xv=(*(cpcifar10Data["train-data"])).block(49000,0,1000,3072);
    MatrixN yv=(*(cpcifar10Data["train-labels"])).block(49000,0,1000,1);
    MatrixN Xt=*(cpcifar10Data["test-data"]);
    MatrixN yt=*(cpcifar10Data["test-labels"]);

    json jo(R"({"verbose":true,"shuffle":true})"_json);
    jo["lr_decay"]=(floatN)1.0;
    jo["batch_size"]=50;

    json j_opt(R"({"type":"Adam"})"_json);
    json j_loss(R"({"type":"CrossEntropy"})"_json);

    Optimizer *pOpt=optimizerFactory("Adam", j_opt);
    Loss *pLoss=lossFactory("SparseCategoricalCrossEntropy", j_loss);

    bool autoOpt=false;

    floatN regularization, regularization_decay, learning_rate, lr_decay;
    learning_rate=1.e-2;
    regularization=1.e-4;
    lr_decay=0.9;
    regularization_decay=0.88;
    if (autoOpt) {
        vector<floatN> regi{1e-3,1e-4,1e-5,1e-6,1e-7}; // -> 1e-5
        vector<floatN> learni{5e-2,1e-2,5e-3,1e-3}; // -> 1e-2
        //vector<floatN> regi{1e-1,5e-2,1e-2,5e-3,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8}; // -> 2e-5 (2nd: 4e-5, 3rd 2e-5)
        //vector<floatN> learni{1e-2,5e-3,1e-3}; // -> 1e-2 (2nd: 6e-3, 3rd 3e-3)
        jo["epochs"]=(floatN)0.1;
        floatN cmAcc=0.0, cAcc;
        for (auto learn : learni) {
            j_opt["learning_rate"]=learn;
            pOpt->updateOptimizerParameters(j_opt);
            for (auto reg : regi) {
                jo["regularization"]=reg;
                t_cppl OptimizerState{};
                cAcc=evalMultilayer(jo, X, y, Xv, yv, Xt, yt, pOpt, &OptimizerState, pLoss, true, true, mode);
                cppl_delete(&OptimizerState);
                if (cAcc > cmAcc) {
                    regularization=reg;
                    learning_rate=learn;
                    cmAcc=cAcc;
                    cerr << green << "Best: Acc:" << cmAcc << ", Reg:" << regularization << ", Learn:" << learning_rate << def << endl;
                } else {
                    cerr << red << "      Acc:" << cAcc << ", Reg:" << reg << ", Learn:" << learn << def << endl;
                }
            }
        }
        cerr << endl << green << "Starting training with: Acc:" << cmAcc << ", Reg:" << regularization << ", Learn:" << learning_rate << def << endl;
    }

    j_opt["learning_rate"]=learning_rate;
    pOpt->updateOptimizerParameters(j_opt);
    jo["regularization"]=regularization;
    jo["lr_decay"]=(floatN)lr_decay;
    jo["regularization_decay"]=(floatN)regularization_decay;
    jo["epochs"]=(floatN)200.0;
    t_cppl OptimizerState{};
    evalMultilayer(jo, X, y, Xv, yv, Xt, yt, pOpt, &OptimizerState, pLoss, true, true, mode);

    for (auto it : cpcifar10Data) {
         free(it.second);
         it.second=nullptr;
    }
    delete pOpt;
    cppl_delete(&OptimizerState);
    delete pLoss;
    return 0;
}

 /*    int id=15;
     cerr << "Class: " << y(id,0) << "=" << classes[y(id,0)] << endl;
     for (int i=0; i<3072; i++) {
         if (i%32==0) cerr<<endl;
         if (i%(32*32)==0) cerr<< "-------------" << endl;
         floatN p=X(id,i);
         if (p<0.25) cerr << " ";
         else if (p<0.5) cerr << ".";
         else if (p<0.75) cerr << "o";
         else cerr << "#";
     }
     cerr << endl << "=====================================================" << endl;

     id=20;
     cerr << "Class: " << y(id,0) << "=" << classes[y(id,0)] << endl;
     for (int i=0; i<3072; i++) {
         if (i%32==0) cerr<<endl;
         if (i%(32*32)==0) cerr<< "-------------" << endl;
         floatN p=X(id,i);
         if (p<0.25) cerr << " ";
         else if (p<0.5) cerr << ".";
         else if (p<0.75) cerr << "o";
         else cerr << "#";
     }
     cerr << endl << "=====================================================" << endl;

     id=25;
     cerr << "Class: " << y(id,0) << "=" << classes[y(id,0)] << endl;
     for (int i=0; i<3072; i++) {
         if (i%32==0) cerr<<endl;
         if (i%(32*32)==0) cerr<< "-------------" << endl;
         floatN p=X(id,i);
         if (p<0.25) cerr << " ";
         else if (p<0.5) cerr << ".";
         else if (p<0.75) cerr << "o";
         else cerr << "#";
     }
     cerr << endl << "=====================================================" << endl;
     for (int i=0; i<3072; i++) {
      cerr << X(id,i) << ", ";
     }
     cerr << endl;
*/
