#include <iostream>
#include <string>
#include <H5Cpp.h>
#include <H5File.h>
#include <H5DataSet.h>

#include "cp-neural.h"

typedef struct {
    int index;               // index of the current object
} iter_info;

herr_t get_all_groups(hid_t loc_id, const char *name, void *opdata);

H5::H5File *cp_Mnist_pfile;
map<string, MatrixN *> cp_Mnist_data;

herr_t get_all_groups(hid_t loc_id, const char *name, void *opdata)
{
    iter_info *info=(iter_info *)opdata;

    // Here you can do whatever with the name...
    cout << "Name : " << name << endl;

    // you can use this call to select just the groups
    // H5G_LINK    0  Object is a symbolic link.
    // H5G_GROUP   1  Object is a group.
    // H5G_DATASET 2  Object is a dataset.
    // H5G_TYPE    3  Object is a named datatype.
    int obj_type = H5Gget_objtype_by_idx(loc_id, info->index);
    if(obj_type == H5G_GROUP)
	   cout << "        is a group" << endl;
    if (obj_type == H5G_DATASET) {
        cout << " is a dataset" << endl;
        H5::DataSet dataset = cp_Mnist_pfile->openDataSet(name);
        /*
         * Get filespace for rank and dimension
         */
        H5::DataSpace filespace = dataset.getSpace();
        /*
         * Get number of dimensions in the file dataspace
         */
        int rank = filespace.getSimpleExtentNdims();
        cout << "rank: " << rank << endl;
        /*
         * Get and print the dimension sizes of the file dataspace
         */
        hsize_t dims[2];    // dataset dimensions
        rank = filespace.getSimpleExtentDims( dims );
        cout << "dataset rank = " << rank << ", dimensions; ";
        for (int j=0; j<rank; j++) {
            cout << (unsigned long)(dims[j]);
            if (j<rank-1) cout << " x ";
        }
        cout << endl;
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
             cout << "float data" << endl;
             auto floatType = dataset.getFloatType();

             size_t byteSize = floatType.getSize();

             if(byteSize == 4)
             {
                 cout << "4-byte float data" << endl;
                  // use PredType::NATIVE_FLOAT to write
                  int *pf = (int *)malloc(sizeof(float) * dims[0] * dims[1]);
                  if (pf) {
                      dataset.read(pf, H5::PredType::NATIVE_FLOAT, mspace1, filespace );
                      for (unsigned int p=0; p<2; p++) {
                          for (unsigned int jy=0; jy<28; jy++) {
                              for (unsigned int jx=0; jx<28; jx++) { // 784=28*28
                                  float pt = pf[p*784+jy*28+jx];
                                  if (pt>0.5) cout << "*";
                                  else cout << " ";
                              }
                              cout << "|"<< endl;
                          }
                          cout << endl;
                      }
                      free(pf);
                  }
             }
             else if(byteSize == 8)
             {
                 cout << "8-byte double float data" << endl;
                  // use PredType::NATIVE_DOUBLE to write
             }
         } else if (dataClass == H5T_INTEGER) {
             cout << "Integer data" << endl;
             int *pi = (int *)malloc(sizeof(int) * dims[0]);
             if (pi) {
                 dataset.read(pi, H5::PredType::NATIVE_INT, mspace1, filespace );
                 for (unsigned int j=0; j<10; j++) cout << pi[j];
                 cout << endl;
                 free(pi);
             }
         }

/*        int data_out[NX][NY];  // buffer for dataset to be read
        dataset.read( data_out, PredType::NATIVE_INT, mspace1, filespace );
        cout << "\n";
        cout << "Dataset: \n";
        for (j = 0; j < dims[0]; j++)
        {
            cout dims[1]
//            for (i = 0; i < dims[1]; i++)
//            cout << data_out[j][i] << " ";
//            cout << endl;
        }
*/
    }
    (info->index)++;
    return 0;
 }

bool  getMnistData(string filepath) {
    H5::H5File fmn((H5std_string)filepath, H5F_ACC_RDONLY);
    cp_Mnist_pfile=&fmn;
    int nr=fmn.getNumObjs();
    cout << nr << endl;
    iter_info info;
    info.index=0;
    fmn.iterateElems("/", NULL, get_all_groups, &info);
    return true;
}

 int main() {
     getMnistData("mnist.hdf5");

     return 0;
 }
