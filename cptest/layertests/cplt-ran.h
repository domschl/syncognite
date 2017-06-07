#ifndef _CPLT_RAN_H
#define _CPLT_RAN_H

#include "../testneural.h"


bool testRAN(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "RAN Layer: " << def << endl;
	// Numerical gradient
    // RAN
	int ranN = 4;
    int ranT = 1; // N=4, D=5, H=6, T=7
	MatrixN xran(ranN, 5 * ranT);
	t_cppl lsstates;
	MatrixN cl0(ranN, 6);
	xran.setRandom();
	cl0.setRandom();
	lsstates["ran-c"] = &cl0;
	// lsstates["ran-h"] = &cl0;
	//                                     D,T
	RAN ran(R"({"name":"ran","inputShape":[5,1],"H":6,"N":4,"noVectorizationTests":true,"nocupdate":true})"_json);
	bool res=ran.selfTest(xran, &lsstates, CP_DEFAULT_NUM_H, CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("RAN", "Numerical gradient", res, "");
	if (!res) bOk = false;

	return bOk;
}

#endif
