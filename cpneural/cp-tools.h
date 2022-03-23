#ifndef _CP_TOOLS_H
#define _CP_TOOLS_H

#include "cp-neural.h"

// for cpInitCompute():
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

//void cppl_delete(t_cppl p);
void cppl_delete(t_cppl *p) {
    int nr=0;
    if (p->size()==0) {
        return;
    }
    for (auto it : *p) {
        if (it.second != nullptr) delete it.second;
        (*p)[it.first]=nullptr;
        ++nr;
    }
    p->erase(p->begin(),p->end());
}

void cppl_set(t_cppl *p, string key, MatrixN *val) {
    auto it=p->find(key);
    if (it != p->end()) {
        cerr << "MEM! Override condition for " << key << " update prevented, freeing previous pointer..." << endl;
        delete it->second;
    }
    (*p)[key]=val;
}

void cppl_update(t_cppl *p, string key, MatrixN *val) {
    if (p->find(key)==p->end()) {
        MatrixN *pm=new MatrixN(*val);
        cppl_set(p, key, pm);
    } else {
        *((*p)[key])=*val;
    }
}

void cppl_remove(t_cppl *p, string key) {
    auto it=p->find(key);
    if (it!=p->end()) {
        if (it->second != nullptr) delete it->second;
        p->erase(it);
    }
}

void cppl_copy(t_cppl *src, t_cppl*dest) {
    dest->clear();
    for (auto st : *src) {
        cppl_set(dest, st.first, new MatrixN(*st.second));
    }
}

void mlPush(string prefix, t_cppl *src, t_cppl *dst) {
    if (dst!=nullptr) {
        for (auto pi : *src) {
            cppl_set(dst, prefix+"-"+pi.first, pi.second);
        }
    } else {
        cppl_delete(src);
    }
}

void mlPop(string prefix, t_cppl *src, t_cppl *dst) {
    for (auto ci : *src) {
        if (ci.first.substr(0,prefix.size()+1)==prefix+"-") {
            cppl_set(dst, ci.first.substr(prefix.size()+1), ci.second);
            //src->erase(ci.first); // XXX for rnn-ho inits! DANGEROUS!
        }
    }
}

// XXX: dubious:
void mlPopX(string prefix, t_cppl *src, t_cppl *dst) {
    for (auto ci=src->cbegin(); ci!=src->cend(); ci++) {
        if (ci->first.substr(0,prefix.size()+1)==prefix+"-") {
            cppl_set(dst, ci->first.substr(prefix.size()+1), ci->second);
            src->erase(ci); // Did not work
        }
    }
}

vector<unsigned int> shape(const MatrixN& m) {
    vector<unsigned int> s(2);
    s[0]=(unsigned int)(m.rows());
    s[1]=(unsigned int)(m.cols());
    return s;
}

bool matCompare(MatrixN& m0, MatrixN& m1, string msg="", floatN eps=1.e-6) {
    if (m0.cols() != m1.cols() || m0.rows() != m1.rows()) {
        cerr << msg << ": Incompatible shapes " << shape(m0) << "!=" << shape(m1) << endl;
        return false;
    }
    MatrixN d = m0 - m1;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {
        if (msg!="") cerr << msg << " err=" << dif << endl;
        return true;
    } else {
        if (msg!="") {
            //IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
            //cerr << msg << " m0:" << endl << m0.format(CleanFmt) << endl;
            //cerr << msg << " m1:" << endl << m1.format(CleanFmt) << endl;
            cerr << "err=" << dif << endl;
        }
        return false;
    }
}

void peekMat(const string label, const MatrixN& m) {
    cerr << label << " ";
    if (m.size()<10) cerr << m << endl;
    else {
        for (int j=0; j<m.size(); j++) {
            if (j<4 || m.size()-j < 4) cerr << m(j) << " ";
            else if (j==4) cerr << " ... ";
        }
        cerr << endl;
    }
}

#define MAX_NUMTHREADS 256
#define MAX_CPUTHREADS 256

MatrixN matmul(MatrixN *a, MatrixN *b, int contextId, bool verbose = false) {
    Timer t, t1;
    t.startWall();
    MatrixN y = *a * (*b);
    if (verbose)
        cerr << "Eigen matmul " << shape(*a) << shape(*b) << "->" << t.stopWallMicro() << endl;
    return y;
}

int cpNumEigenThreads=1;
int cpNumCpuThreads=1;

int cpGetNumEigenThreads() {
    return cpNumEigenThreads;
}
int cpGetNumCpuThreads() {
    return cpNumCpuThreads;
}

bool cpInitCompute(string name, CpParams* poptions=nullptr, int verbose=1) {
    CpParams cp;
    string options="";
    struct passwd *pw = getpwuid(getuid());
    const char *homedir = pw->pw_dir;
    string conffile = string(homedir) + "/.syncognite";
    std::ifstream cfile(conffile);
    if (cfile.is_open()) {
        string line;
        string conf;
        while (std::getline(cfile,line)) {
            conf+=line;
        }
        cp.setString(conf);
        cfile.close();
    } else {
        if (verbose>0) cerr << "New configuration, '" << conffile << "' not found." << endl;
    }
    cpNumEigenThreads=cp.getPar("NumEigenThreads", 1);
    int numHWThreads=std::thread::hardware_concurrency();
    cpNumCpuThreads=cp.getPar("NumCpuThreads", numHWThreads);
    if (cpNumCpuThreads > MAX_CPUTHREADS) {
        cpNumCpuThreads=MAX_CPUTHREADS;
    }
    if (poptions!=nullptr) {
        *poptions=cp;
    }

    // Eigen::initParallel();
    Eigen::setNbThreads(cpNumEigenThreads);

    #ifdef USE_FLOAT
    options+="FLOAT ";
    #endif
    #ifdef USE_AVX
    options+="AVX ";
    #endif
    #ifdef USE_SSE2
    options+="SSE2 ";
    #endif
    #ifdef USE_SSE4
    options+="SSE4 ";
    #endif
    #ifdef USE_FMA
    options += "FMA ";
    #endif
    #ifdef USE_OPENMP
    options += "OPENMP ";
    #endif

    std::ofstream c2file(conffile);
    if (c2file.is_open()) {
        string line;
        line=cp.getString();
        c2file << line << endl;
        c2file.close();
    }
    if (cpNumCpuThreads > MAX_NUMTHREADS) {
        cerr << "Number of CPU threads must not be > " << MAX_NUMTHREADS << endl;
        cerr << "INVALID CONFIGURATION" << endl;
        return false;
    }
    if (verbose>0) {
        cerr << "Compile-time options: " << options << endl;
        cerr << "Eigen is using:      " << cpNumEigenThreads << " threads." << endl;
        cerr << "CpuPool is using:    " << cpNumCpuThreads << " threads." << endl;
    }
    return true;
}

#endif
