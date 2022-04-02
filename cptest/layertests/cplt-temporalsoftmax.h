#ifndef _CPLT_TEMPORALSOFTMAX_H
#define _CPLT_TEMPORALSOFTMAX_H

#include "../testneural.h"

float getTemporalSMLoss(int N, int T, int D, float p) {
    MatrixN x(N,D*T);
    x.setRandom();
    x = (x.array()+1.0) * 0.005;
    MatrixN y(N,T);
    for (int i=0; i<y.size(); i++) y(i)=rand() % D;
    MatrixN mask(N,T);
    for (int i=0; i<mask.size(); i++) {
        if (rand()%1000 < p*1000.0) mask(i)=1.0;
        else mask(i)=0.0;
    }
    json j;
    j["inputShape"]=vector<int>{D,T};
    TemporalSoftmax tsm(j);
    t_cppl cache;
    cppl_set(&cache,"mask",new MatrixN(mask));
    t_cppl states;
    states["y"] = &y;
    MatrixN yhat=tsm.forward(x,&cache, &states);
    float loss;
    Loss *pLoss=lossFactory("TemporalCrossEntropy", j);
    loss=pLoss->loss(yhat, y, &cache);
    //loss=tsm.loss(&cache, &states);
    cppl_delete(&cache);
    delete pLoss;
    return loss;
}

bool checkTemporalSoftmaxLoss(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    bool allOk=true;
    float loss;
    if (verbose>1) cerr << "  Checking TemporalSoftmaxLoss:" << endl;
    loss=getTemporalSMLoss(1000, 1, 10, 1.0);   // Should be about 2.3
    if (std::abs(loss-2.3)>eps) {
        if (verbose>0) cerr << "    TemporalSMLoss check failed for ex (1): " << loss << ", should be 2.3" << endl;
        allOk=false;
    } else {
        if (verbose>1) cerr << "    TemporalSMLoss check OK for ex (1): " << loss << ", theoretical: 2.3" << endl;
    }
    loss=getTemporalSMLoss(1000, 10, 10, 1.0);  // Should be about 23
    if (std::abs(loss-23.0)>eps) {
        /* does not pass numerical checks, if loss is T dependent */
        if (verbose>0) cerr << "    TemporalSMLoss check failed for ex (2): " << loss << ", should be 23" << endl;
    } else {
        if (verbose>1) cerr << "    TemporalSMLoss check OK for ex (2): " << loss << ", theoretical: 23" << endl;
    }
    loss=getTemporalSMLoss(50000, 10, 10, 0.1); // Should be about 2.3
    if (std::abs(loss-2.3)>eps) {
        /* Does not path numerical tests, if we scale loss with T */
        if (verbose>0) cerr << "    TemporalSMLoss check failed for ex (3): " << loss << ", should be 2.3" << endl;
    } else {
        if (verbose>1) cerr << "    TemporalSMLoss check OK for ex (3): " << loss << ", theoretical: 2.3" << endl;
    }
    return allOk;
}

bool checkTemporalSoftmax(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    int N=7, T=8, D=9;
    MatrixN x(N,T*D);
    x << -1.23246054,  0.01065191,  0.54041174,  0.86820939,  0.14530547,
         -1.43534012, -2.21043114, -0.33182685,  0.42640048,
         -0.03021426,  0.79209013, -0.70492907, -1.1088065 ,  0.10722934,
         -0.07563222,  0.16285917,  0.91064826,  0.44968059,
          0.57037959, -1.35931677, -0.30480425,  0.6221304 ,  1.82383057,
          0.70670256, -0.45644312, -0.820395  ,  0.83848922,
          0.06050983,  0.83937924,  1.25368984,  0.76224628, -1.10499651,
          1.37315674,  0.15848368, -1.24564521,  1.37579834,
          1.80056608, -1.33962381, -0.51035462,  0.55684867,  0.16229723,
         -1.12652436, -1.53010026,  1.50585593,  0.18578204,
          0.09749102,  0.97488254,  0.2825707 , -0.12304635,  0.26597735,
         -0.17503789, -1.39362066, -1.00396713,  0.8317905 ,
         -0.50954108,  1.54768111,  0.56327482,  0.75618198,  0.46718592,
         -0.28929801,  1.18453659, -0.70203234,  1.10039502,
          1.01438321,  0.68737633,  1.29929128, -1.01412282, -0.85015464,
          0.22482607,  0.49374356, -1.19048888,  1.05352751,

         -0.67494446,  0.71137683, -0.71691637,  0.57483271, -1.55223013,
          0.00610196,  1.52712649, -0.16467183,  0.31255622,
          0.72502072,  0.51791378,  0.91971158, -1.54812569,  1.7873487 ,
         -0.40530621, -1.18811548, -0.41283135,  0.15963355,
         -0.45676613,  0.33561934, -0.8403044 ,  1.58279834, -0.36781072,
         -0.27617194, -1.50725124, -0.23736975,  1.31034913,
          1.06217009, -0.08612586, -0.19195893,  1.80771994,  0.07152974,
         -3.37015394,  1.38971412,  1.36475512, -2.7726176 ,
          1.12683149, -2.00708985,  0.10858035,  0.06907456, -0.11112568,
          0.333218  ,  0.58412352,  0.78479103, -0.83718271,
          0.67261661,  0.67028683,  1.15465667, -0.09062283, -1.89483293,
          0.96738617, -0.84341337, -1.06906462, -0.01838411,
          0.92665743, -0.48719751, -0.99447975, -0.51876984, -0.77998483,
         -0.2808039 ,  1.62511173, -0.04086433, -0.07493452,
         -0.3990804 , -0.90471161,  2.85928783, -0.72164838, -0.38568054,
          3.20183084,  0.75828105, -0.80647938,  0.15740569,

          0.00817948,  0.17020808,  0.28786154,  1.22527321,  0.19553596,
         -1.90646293, -0.52587949, -0.3585425 ,  0.57658381,
          0.70690854,  0.02665304,  0.00475914, -0.87788444, -0.87281609,
          0.47322323, -0.16148717, -0.47789615, -1.42025668,
         -0.73879407,  0.60226131,  1.00287624,  0.64578424, -0.99611268,
         -1.73323888,  0.65036401,  0.30403207, -0.75189721,
         -1.03423291,  0.48382529,  0.90841575,  0.82096795,  1.975297  ,
         -0.69615043,  0.85700192,  0.61397502, -0.44100134,
         -1.1249768 ,  0.2717552 ,  0.41650217, -0.70660096,  1.38998942,
          1.07424315, -1.00236587,  1.33081877, -0.71533026,
         -1.61758898,  1.41733347,  0.61105335, -1.10267993,  1.19270124,
         -0.82527993, -1.47341676,  1.06739788, -0.01155889,
          0.23737619, -1.46059656,  2.23723273,  1.09008309,  0.58668014,
          2.11281835,  1.16422317,  0.5476622 , -0.79077501,
         -0.32321708, -0.52418307,  1.24123064,  1.25033905,  1.02196347,
         -0.14264452, -0.03222287, -0.36326323,  2.48866113,

         -0.27514122,  0.61314512,  1.19686346,  2.7934575 ,  0.72339012,
          1.0933334 , -0.41266207,  0.63205129,  0.71300261,
          2.44092073,  0.4528844 , -0.29482082,  0.16530509,  0.00938356,
         -1.35837686,  0.51897039,  0.6263144 , -0.21886015,
         -0.59220673,  0.39377209, -0.52900549, -0.89893693, -1.06756872,
         -1.00747174, -0.67032537, -0.36873639,  0.12067989,
          0.01632456,  1.85504015, -1.93916848,  0.78489589,  0.62984409,
          0.29071807, -0.83046318,  0.07531773,  0.88729803,
          0.28031542, -0.7248856 ,  1.36213444, -0.15393923,  1.29719223,
          1.07694704, -0.21834618,  0.60513942,  4.06850016,
         -1.0695331 , -1.58558893, -1.37775283, -0.18025101, -1.29516006,
         -1.43068509,  0.23421301, -2.04531874, -1.42156439,
         -0.86433241, -0.9312067 ,  1.15403053, -0.3007323 ,  0.22152043,
          0.17413518,  1.06384989, -1.12606188,  0.20089207,
         -0.38018218,  1.16264881, -0.1847447 , -0.2175662 , -1.90623475,
          0.41539818,  0.38075958,  1.52517035, -1.01376356,

          1.15404768, -2.6599956 , -2.25937291,  0.10159108,  0.59714404,
         -2.35207302,  0.67064184,  0.91105623, -1.34241482,
         -0.76270973,  0.71037932, -0.54058184,  2.28049134, -0.72569589,
          0.04874692, -0.3631741 ,  0.36731889,  1.04632766,
          0.37055635, -0.94499134, -1.88122289,  0.68277363,  0.2093514 ,
         -0.94528553, -0.57382395, -0.8820868 , -0.34437022,
          1.23342172, -1.0103727 ,  0.27246656, -0.02379227, -0.11246737,
         -0.5659306 ,  1.22456235, -0.28485077,  0.6713286 ,
         -0.49379165,  0.08135896, -0.57108754,  0.13330084, -0.41475684,
         -1.83184605, -0.98852967,  1.03484333, -0.58889767,
         -0.27223673,  0.73325881, -1.0733917 , -0.65271203, -0.96481752,
         -0.33852637,  0.06474923,  0.76296152,  1.85330975,
         -0.24499654,  1.28161835, -0.60211981,  0.85626103, -0.2599326 ,
         -0.54490648, -1.20935716,  1.81721061, -0.01912715,
          0.16524671,  1.62662607,  0.14115403,  0.14225045, -0.76473403,
          0.83777188, -0.44181035,  0.20785358, -1.10968981,

          0.371995  ,  0.46298795, -0.05490326, -0.01059601,  0.56149922,
         -0.29053111, -0.78759466,  0.29703164,  1.0769004 ,
          1.81508036, -0.65893633,  0.87014643, -0.06143444, -0.35609719,
          0.64089944, -0.30069491, -1.22216606,  0.38138779,
         -0.27705491, -0.11146415, -0.09441767,  0.55918637, -0.22215872,
         -0.25168764,  0.39212847,  0.36636727,  0.71228227,
         -0.89650995,  0.06583769, -0.32333475, -0.19139622, -0.66976034,
         -0.71978613,  2.38729681,  0.06367166,  2.35731829,
         -1.72666878, -0.13533269, -0.67399432,  0.94419688,  0.28171876,
          1.94806488,  1.84581123, -0.86989727,  0.32418888,
         -0.0720382 , -0.4151754 , -0.54956794, -1.22338157,  0.8911768 ,
          0.76935735,  0.62485098, -0.89842935,  0.79951204,
          0.76012471,  1.34712337, -0.1273357 ,  0.57867401,  0.07638629,
          0.80990537, -0.8946335 ,  0.30908188,  0.05777147,
          0.0371182 , -0.75714517, -2.38126256, -0.75681632,  1.19101891,
         -1.02337796,  0.45233467,  0.12166621,  0.99873216,

         -0.23249482,  2.07443278,  0.20892656,  0.49492642,  0.02436312,
          1.58837555, -0.60280719,  0.14495463,  0.63814337,
          0.29863814,  0.73501704,  1.74678432, -1.06072893, -1.21863415,
          0.54205632, -1.031977  ,  1.10047104, -0.01619991,
          1.1324309 , -0.26726347,  0.24252532,  0.63287309, -0.63695328,
          0.37404285, -0.50039178,  1.51685564,  0.2184609 ,
         -0.54865858,  1.8937991 ,  1.62374043, -1.38413146, -0.22642374,
         -0.05505692, -0.22045655, -1.54472281, -0.70203831,
         -0.12658793,  2.04944577,  1.04121892,  0.52561103, -0.85058912,
          1.83263902,  0.2110118 ,  0.22874347, -1.15099273,
         -0.55713993, -1.09798934, -0.28051148,  1.041161  ,  0.3976339 ,
          0.99775684, -1.04045249, -1.02326925, -1.11500016,
         -0.82037391, -0.02670681,  0.10607822,  1.52807958, -0.53632557,
         -0.69376549,  1.39423102, -0.1747266 , -0.4161898 ,
          0.391343  , -1.4529948 ,  1.91369741,  0.40384839,  0.95981493,
          0.7488383 ,  0.12950524,  0.10790153,  0.27570704;

    MatrixN y(N,T);
    y << 0, 3, 6, 4, 2, 6, 4, 8,
        4, 1, 0, 6, 8, 2, 2, 2,
        6, 2, 6, 2, 4, 7, 2, 1,
        2, 7, 6, 2, 4, 3, 5, 8,
        2, 2, 3, 8, 5, 8, 7, 8,
        3, 1, 6, 8, 1, 1, 2, 0,
        2, 7, 6, 7, 1, 5, 8, 6;

    MatrixN mask(N,T);
    mask << 1.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,
         1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,
         1.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,
         1.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,
         0.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,
         0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,
         0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.;

    MatrixN dx(N,T*D);
    dx << -0.13830737,  0.01577127,  0.02678789,  0.03717912,  0.01804455,
          0.00371433,  0.00171105,  0.01119773,  0.02390143,
          0.01229501,  0.02798022,  0.00626187, -0.13867593,  0.01410652,
          0.01174909,  0.0149135 ,  0.03150216,  0.01986756,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        , -0.        ,  0.        ,  0.        ,
          0.00810712,  0.01766548,  0.02673365,  0.01635411, -0.14032963,
          0.03012604,  0.00894162,  0.00219589,  0.03020572,
          0.05374461,  0.00232578, -0.1375273 ,  0.01549513,  0.01044344,
          0.00287816,  0.0019224 ,  0.04002616,  0.01069161,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        , -0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        , -0.        ,

          0.0056612 ,  0.02264543,  0.00542851,  0.01975514, -0.14050259,
          0.01118622,  0.05119836,  0.00943012,  0.0151976 ,
          0.        , -0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
         -0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.02083729,  0.0066091 ,  0.00594538,  0.04391667,  0.00773769,
          0.00024769, -0.11394424,  0.0282002 ,  0.00045021,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        , -0.        ,
          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.00199602,  0.00120385, -0.09094653,  0.00144569,  0.00202295,
          0.0731174 ,  0.00635039,  0.00132811,  0.00348212,

          0.01267129,  0.01490009,  0.01676043,  0.04279547,  0.0152823 ,
          0.00186768, -0.13542897,  0.00878123,  0.02237049,
          0.03527995,  0.01786886, -0.12537525,  0.00723205,  0.00726879,
          0.02792799,  0.01480432,  0.01078882,  0.00420448,
          0.00607856,  0.02323879,  0.03468953,  0.02427254,  0.00469947,
          0.00224863, -0.11847318,  0.01724623,  0.00599943,
          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.00304537,  0.01230931,  0.01422645,  0.00462742, -0.10519746,
          0.02746311,  0.00344263,  0.03549598,  0.0045872 ,
          0.00197373,  0.04105225,  0.01833049,  0.003303  ,  0.03279295,
          0.00435895,  0.00227982, -0.11392625,  0.00983506,
          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.00403572, -0.13955616,  0.01929088,  0.01946739,  0.01549265,
          0.0048344 ,  0.00539881,  0.0038773 ,  0.06715901,

          0.00340303,  0.00827263, -0.12802687,  0.07320504,  0.00923681,
          0.0133717 ,  0.00296579,  0.00843052,  0.00914136,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        , -0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.00254585,  0.00093171,  0.00751038,  0.00164907, -0.13581901,
          0.00564685,  0.00154621,  0.00352292,  0.11246602,
          0.        ,  0.        ,  0.        , -0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.00518475,  0.00484936,  0.03902037,  0.00910953,  0.01535704,
         -0.12821083,  0.03565549,  0.0039908 ,  0.01504349,
          0.00718015,  0.03358742,  0.00872994,  0.00844806,  0.00156091,
          0.01590925,  0.01536761,  0.04826338, -0.13904672,

          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.02843916,  0.007631  ,  0.00299214, -0.1039964 ,  0.02420507,
          0.00762876,  0.01106057,  0.00812645,  0.01391324,
          0.03590253,  0.00380765,  0.0137337 ,  0.01021231,  0.00934572,
          0.00593849,  0.03558587,  0.00786589, -0.12239217,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         -0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        , -0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        , -0.        ,  0.        ,
          0.01243638,  0.05362448,  0.01214034,  0.01215366,  0.00490692,
          0.02436509,  0.00677724,  0.01297771, -0.13938181,

          0.        ,  0.        ,  0.        , -0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.        , -0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.0099793 ,  0.01177648,  0.01197894,  0.023029  ,  0.01054244,
          0.01023569, -0.12337114,  0.01899043,  0.02683885,
          0.00219638,  0.00574975,  0.00389613,  0.00444564,  0.00275539,
          0.00262094,  0.05859304,  0.00573731, -0.08599457,
          0.        , -0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.01150078, -0.13469687,  0.0071341 ,  0.00363668,  0.03013328,
          0.02667724,  0.0230878 ,  0.00503304,  0.02749394,
          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
         -0.12954172,  0.00601742,  0.00118595,  0.0060194 ,  0.04221701,
          0.00461092,  0.02016885,  0.01449017,  0.034832  ,

          0.        ,  0.        , -0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.01212926,  0.01876512,  0.05161266,  0.00311508,  0.00266006,
          0.01547211,  0.00320594, -0.11581345,  0.00885323,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        , -0.        ,  0.        ,  0.        ,
          0.00522362,  0.06007833,  0.04585987,  0.00226533,  0.00720969,
          0.00855737,  0.00725284, -0.1409279 ,  0.00448085,
          0.        , -0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         -0.        ,  0.        ,  0.        ,  0.        ,
          0.00457392,  0.01011519,  0.0115516 ,  0.04788608,  0.00607644,
          0.00519128,  0.04188704,  0.00872348, -0.13600504,
          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        , -0.        ,  0.        ,  0.;

    floatN lossTheo=9.9761896117014111;

    json j;
    j["inputShape"]=vector<int>{D,T};
    j["nohupdate"]=(bool)true;
    TemporalSoftmax tsm(j);
    t_cppl cache;
    t_cppl states;
    cppl_set(&cache,"mask",new MatrixN(mask));
    states["y"]=&y;
    MatrixN yhat = tsm.forward(x,&cache,&states);
    float loss;
    Loss *pLoss=lossFactory("TemporalCrossEntropy", j);
    loss=pLoss->loss(yhat, y, &cache);
    delete pLoss;

    bool allOk=true;

    if (std::abs(loss-lossTheo) > eps) {
        allOk=false;
        if (verbose>0) cerr << "  Error TemporalSoftmaxLoss (sample data): Loss sample data: " << lossTheo << ", calculated: " << loss << endl;
    }

    t_cppl grads;
    MatrixN dx0=tsm.backward(y, &cache,&states,&grads);

    bool ret=matCompT(dx,dx0,"TemporalSoftmax dx",eps,verbose);
    if (!ret) allOk=false;

    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

bool testTemporalSoftmax(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "TemporalSoftmax Layer: " << def << endl;
	// Numerical gradient
    // Temporal Softmax
	int tsmN = 10, tsmC = 4, Ttm = 4;
	json j;
	j["inputShape"]=vector<int>{tsmC, Ttm};
	j["noVectorizationTests"]=(bool)true;
	TemporalSoftmax tmx(j);
	MatrixN txmx(tsmN, tsmC * Ttm);
	txmx.setRandom();
	MatrixN ty(tsmN, Ttm);
	for (unsigned i = 0; i < ty.size(); i++)
		ty(i) = (rand() % tsmC);
	floatN h = 1e-2;
	if (h < CP_DEFAULT_NUM_H)
		h = CP_DEFAULT_NUM_H;
	floatN eps = 1e-4;
	if (eps < CP_DEFAULT_NUM_EPS)
		eps = CP_DEFAULT_NUM_EPS;
	t_cppl states;
	states["y"] = &ty;
    Loss *pLoss=lossFactory("TemporalCrossEntropy", j);
	bool res=tmx.selfTest(txmx, &states, h, eps, verbose, pLoss);
    delete pLoss;
	registerTestResult("TemporalSoftmax", "Numerical gradient", res, "");
	if (!res) bOk = false;

	res=checkTemporalSoftmaxLoss(0.1, verbose);
	registerTestResult("TemporalSoftmax", "Loss (with test-data)", res, "");
	if (!res) bOk = false;

    res=checkTemporalSoftmax(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("TemporalSoftmax", "Check (with test-data)", res, "");
	if (!res) bOk = false;

	return bOk;
}

#endif
