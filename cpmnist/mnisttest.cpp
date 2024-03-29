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
	cerr << name << " ";
	int obj_type = H5Gget_objtype_by_idx(loc_id, info->index);
	if (obj_type == H5G_DATASET) {
		H5::DataSet dataset = cp_Mnist_pfile->openDataSet(name);
		H5::DataSpace filespace = dataset.getSpace();
        int rank; // = filespace.getSimpleExtentNdims();
		hsize_t dims[2];    // dataset dimensions
		rank = filespace.getSimpleExtentDims( dims );
		if (rank==1) dims[1]=1;
		cpMnistData[name] = new MatrixN(dims[0],dims[1]);
		MatrixN *pM = cpMnistData[name];
		H5::DataSpace mspace1(rank, dims);
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
		cerr << "cpMnistData contains already elements, not reloading." << endl;
		return true;
	}
	H5::H5File fmn((H5std_string)filepath, H5F_ACC_RDONLY);
	cp_Mnist_pfile=&fmn;
	//int nr=fmn.getNumObjs();
	//cerr << nr << endl;
	cp_mnist_iter_info info;
	info.index=0;
	cerr << "Reading: ";
	fmn.iterateElems("/", NULL, cp_mnist_get_all_groups, &info);
	cerr << endl;
	return true;
}

void printSamples() {
	vector<int> ins {0,4,16,25,108, 256,777};
	for (auto in : ins) {
		cerr << "-------------------------" << endl;
		cerr << "Index: " << in << endl;
		for (int cy=0; cy<28; cy++) {
			for (int cx=0; cx<28; cx++) {
				floatN pt=(*cpMnistData["x_test"])(in, cy*28+cx);
				if (pt<0.5) cerr << " ";
				else cerr << "*";
			}
			cerr << endl;
		}
		cerr << (*cpMnistData["t_test"])(in,0) << endl;
	}
}

int main(int argc, char *argv[]) {
	if (argc!=2) {
		cerr << "mnisttest <path-mnist.h5-file>" << endl;
		exit(-1);
	}
	getMnistData(argv[1]);
	for (auto it : cpMnistData) {
		cerr << it.first << " " << shape(*(it.second)) << endl;
	}

	// printSamples(); exit(-1);

	cpInitCompute("Mnist");
	registerLayers();

	MatrixN X=*(cpMnistData["x_train"]);
	MatrixN y=*(cpMnistData["t_train"]);
	MatrixN Xv=*(cpMnistData["x_valid"]);
	MatrixN yv=*(cpMnistData["t_valid"]);
	MatrixN Xt=*(cpMnistData["x_test"]);
	MatrixN yt=*(cpMnistData["t_test"]);

	LayerBlockOldStyle lb(R"({"name":"DomsNet","bench":false,"init":"orthonormal"})"_json);

	lb.addLayer("Convolution", "cv1", R"({"inputShape":[1,28,28],"kernel":[48,5,5],"stride":1,"pad":2})",{"input"});
	lb.addLayer("BatchNorm","sb1","{}",{"cv1"});
	lb.addLayer("Relu","rl1","{}",{"sb1"});
	lb.addLayer("Dropout","doc1",R"({"drop":0.8})",{"rl1"});
	lb.addLayer("Convolution", "cv2", R"({"kernel":[48,3,3],"stride":1,"pad":1})",{"doc1"});
	lb.addLayer("Relu","rl2","{}",{"cv2"});
	lb.addLayer("Convolution", "cv3", R"({"kernel":[64,3,3],"stride":2,"pad":1})",{"rl2"});
	lb.addLayer("BatchNorm","sb2","{}",{"cv3"});
	lb.addLayer("Relu","rl3","{}",{"sb2"});
	lb.addLayer("Dropout","doc2",R"({"drop":0.8})",{"rl3"});
	lb.addLayer("Convolution", "cv4", R"({"kernel":[64,3,3],"stride":1,"pad":1})",{"doc2"});
	lb.addLayer("Relu","rl4","{}",{"cv4"});
	lb.addLayer("Convolution", "cv5", R"({"kernel":[128,3,3],"stride":2,"pad":1})",{"rl4"});
	lb.addLayer("BatchNorm","sb3","{}",{"cv5"});
	lb.addLayer("Relu","rl5","{}",{"sb3"});
	lb.addLayer("Dropout","doc3",R"({"drop":0.8})",{"rl5"});
	lb.addLayer("Convolution", "cv6", R"({"kernel":[128,3,3],"stride":1,"pad":1})",{"doc3"});
	lb.addLayer("Relu","rl6","{}",{"cv6"});

	lb.addLayer("Affine","af1",R"({"hidden":1024})",{"rl6"});
	lb.addLayer("BatchNorm","bn1","{}",{"af1"});
	lb.addLayer("Relu","rla1","{}",{"bn1"});
	lb.addLayer("Dropout","do1",R"({"drop":0.7})",{"rla1"});
	lb.addLayer("Affine","af2",R"({"hidden":512})",{"do1"});
	lb.addLayer("BatchNorm","bn2","{}",{"af2"});
	lb.addLayer("Relu","rla2","{}",{"bn2"});
	lb.addLayer("Dropout","do2",R"({"drop":0.7})",{"rla2"});
	lb.addLayer("Affine","af3",R"({"hidden":10})",{"do2"});
	lb.addLayer("Softmax","sm1","{}",{"af3"});

	bool verbose=true;
	if (verbose) cerr << "Checking multi-layer topology..." << endl;
	if (!lb.checkTopology(verbose)) {
		cerr << "Topology-check for MultiLayer: ERROR." << endl;
		exit(-1);
	} else {
		if (verbose) cerr << "Topology-check for MultiLayer: ok." << endl;
	}

	json jo(R"({"verbose":true,"shuffle":true, "lr_decay": 0.9})"_json);
	jo["epochs"]=(floatN)40.0;
	jo["batch_size"]=50;
	jo["regularization"]=(floatN)1e-4;
    jo["regularization_decay"]=(floatN)0.87;

    json j_opt(R"({"name":"Adam","beta1":0.9,"beta2":0.999,"epsilon":1e-8})"_json);
	j_opt["learning_rate"]=(floatN)2e-2;
    json j_loss(R"({"name":"CrossEntropy"})"_json);
    Optimizer *pOptimizer=optimizerFactory("Adam", j_opt);
    t_cppl OptimizerState{};
    Loss *pLoss=lossFactory("SparseCategoricalCrossEntropy", j_loss);

	lb.train(X, y, Xv, yv, pOptimizer, &OptimizerState, pLoss, jo);

    delete pOptimizer;
    cppl_delete(&OptimizerState);
    delete pLoss;

	floatN train_err, val_err, test_err;
	train_err=lb.test(X, y, jo.value("batch_size", 50));
	val_err=lb.test(Xv, yv, jo.value("batch_size", 50));
	test_err=lb.test(Xt, yt, jo.value("batch_size", 50));

	cerr << "Final results on MNIST after " << jo.value("epochs",(floatN)0.0) << " epochs:" << endl;
	cerr << "      Train-error: " << train_err << " train-acc: " << 1.0-train_err << endl;
	cerr << " Validation-error: " << val_err <<   "   val-acc: " << 1.0-val_err << endl;
	cerr << "       Test-error: " << test_err <<  "  test-acc: " << 1.0-test_err << endl;

	for (auto it : cpMnistData) {
		free(it.second);
		it.second=nullptr;
	}
}
