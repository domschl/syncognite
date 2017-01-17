#ifndef _CPL_RNN_H
#define _CPL_RNN_H

#include "../cp-layers.h"


class RNN : public Layer {
private:
    int numGpuThreads;
    int numCpuThreads;
    floatN initfactor;
    int T,D,H,N;
    float maxClip=0.0;
    bool nohupdate;
    void setup(const CpParams& cx) {
        layerName="RNN";
        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL & LayerType::LT_EXTERNALSTATE;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        H=cp.getPar("H",1024);
        // T=cp.getPar("T",3);
        N=cp.getPar("N",1);
        XavierMode inittype=xavierInitType(cp.getPar("init",(string)"standard"));
        initfactor=cp.getPar("initfactor",(floatN)1.0);
        maxClip=cp.getPar("clip",(float)0.0);
        nohupdate=cp.getPar("nohupdate",(bool)false);  // true for auto-diff tests
        // D=inputShapeFlat;
        D=inputShape[0];
        T=inputShape[1];
        //outputShape={T*H};
        outputShape={H,T};

        cppl_set(&params, "ho", new MatrixN(N,H));
        cppl_set(&params, "Wxh", new MatrixN(xavierInit(MatrixN(D,H),inittype,initfactor)));
        cppl_set(&params, "Whh", new MatrixN(xavierInit(MatrixN(H,H),inittype,initfactor)));
        cppl_set(&params, "bh", new MatrixN(xavierInit(MatrixN(1,H),inittype,initfactor)));
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        params["ho"]->setZero();

/*
        params["Wxh"]->setRandom();
        params["Whh"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(inputShapeFlat+H)); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["Wxh"] *= xavier;
        *params["Whh"] *= xavier;
        params["bh"]->setRandom();
        *params["bh"] *= xavier;
*/
        layerInit=true;
    }
public:
    RNN(const CpParams& cx) {
        setup(cx);
    }
    RNN(const string conf) {
        setup(CpParams(conf));
    }
    ~RNN() {
        cppl_delete(&params);
    }
    virtual void genZeroStates(t_cppl* pstates, int N) {
        MatrixN *ph= new MatrixN(N,H);
        ph->setZero();
        cppl_set(pstates, "h", ph);
    }

    virtual MatrixN forward_step(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) {
        // h(t)=tanh(Whh·h(t-1) + Wxh·x(t) + bh)    h(t-1) -> ho,   h(t) -> h
        if (pcache!=nullptr) if (pcache->find("x")==pcache->end()) cppl_set(pcache, "x", new MatrixN(x));
        int N=shape(x)[0];
        MatrixN *ph;
        if (pcache==nullptr || pcache->find("h")==pcache->end()) {
            ph=new MatrixN(N,H);
            ph->setZero();
            if (pcache!=nullptr) cppl_set(pcache,"h",ph);
            else free(ph);
        } else {
            if (pcache==nullptr) cerr << "pcache must not be null in rnn_step_forward" <<endl;
            ph=(*pcache)["h"];
            cppl_set(pcache,"ho",new MatrixN(*ph));
        }
        //cerr << shape(*ph) << shape(*params["Whh"]) << shape(x) << shape(*params["Wxh"]) << endl;
        MatrixN hn = ((*ph * *params["Whh"] + x * *params["Wxh"]).rowwise() + RowVectorN(*params["bh"])).array().tanh();
        *ph=hn;
        return hn;
    }
    MatrixN tensorchunk(const MatrixN& x, vector<int>dims, int b) {
        int A=dims[0];
        //int B=dims[1];
        int C=dims[2];
        MatrixN xi(A,C);
        int a,c;
        for (a=0; a<A; a++) {
            for (c=0; c<C; c++) {
                xi(a,c) = x(a,b*C+c);
            }
        }
        return xi;
    }
    void tensorchunkinsert(MatrixN *ph, MatrixN& ht, vector<int>dims, int b) {
        int A=dims[0];
        //int B=dims[1];
        int C=dims[2];
        int a,c;
        for (a=0; a<A; a++) {
            for (c=0; c<C; c++) {
                (*ph)(a,b*C+c) = ht(a,c);
            }
        }
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (x.cols() != D*T) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x:" << shape(x) << " D*T:" << D*T << ", D:" << D << ", T:" << T << ", H:" << H << endl;
            MatrixN h(0,0);
            return h;
        }
        int N=shape(x)[0];
        MatrixN h0;
        if (pcache!=nullptr) {
            if (pcache->find("x")==pcache->end()) cppl_set(pcache, "x", new MatrixN(x));
        }

        // if (id>0) cerr << "WARNING: rnn layer is not thread-safe, thread id>0! (set optimizer parameter maxthreads=1)" << endl;
        if (pcache!=nullptr && pcache->find("hoi")!=pcache->end()) {
            h0=*((*pcache)["hoi"]); // XXX: UUUUHHH!
        } else {
            h0=*params["ho"];  // XXX: REALLY not!
        }
        MatrixN h(N,T*H);
        h.setZero();
        MatrixN ht=h0;
        MatrixN xi;
        string name;
        for (int t=0; t<T; t++) {
            t_cppl cache{};
            t_cppl states{}; // XXX  NEEDS TO BE FILLED!
            xi=tensorchunk(x,{N,T,D},t);
            cppl_set(&cache,"h",new MatrixN(ht)); // XXX: THIS NEEDS TO BE IN STATES!
            /// XXX: HERE IT GOES! cppl_set(&states,)
            ht = forward_step(xi,&cache,&states, id);
            tensorchunkinsert(&h, ht, {N,T,H}, t);
            name="t"+std::to_string(t);
            mlPush(name,&cache,pcache);
        }
        // If we update ho, auto-differenciation wont work.
        if (pcache!=nullptr && pcache->find("ho")!=pcache->end()) {
            *(*pcache)["ho"]=ht;
        } else {
            if (!nohupdate) *params["ho"]=ht; // XXX WARNING: this makes the whole thing not thread-safe and state-dependant!
        }
        return h;
    }
    virtual MatrixN backward_step(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) {
        if (pcache->find("x") == pcache->end()) {
            cerr << "cache does not contain x -> fatal!" << endl;
        }
        MatrixN x(*(*pcache)["x"]);
        // XXX: TODO: STATES benutzen!
        MatrixN h(*(*pcache)["h"]);
        MatrixN ho(*(*pcache)["ho"]);
        MatrixN Wxh(*params["Wxh"]);
        MatrixN Whh(*params["Whh"]);
        MatrixN bh(*params["bh"]);

        MatrixN hsq = h.array() * h.array();
        MatrixN hone = MatrixN(h);
        hone.setOnes();
        MatrixN t1=(hone-hsq).array() * dchain.array();
        MatrixN t1t=t1.transpose();
        MatrixN dbh=t1.colwise().sum();
        MatrixN dx=(Wxh * t1t).transpose();
        MatrixN dWxh=(t1t * x).transpose();
        MatrixN dh=(Whh * t1t).transpose();
        MatrixN dWhh=ho.transpose() * t1;

        (*pgrads)["Wxh"] = new MatrixN(dWxh);
        (*pgrads)["Whh"] = new MatrixN(dWhh);
        (*pgrads)["bh"] = new MatrixN(dbh);
        (*pgrads)["ho"] = new MatrixN(dh);

        if (maxClip != 0.0) {
            for (int i=0; i<dx.size(); i++) {
                if (dx(i) < -1.0 * maxClip) dx(i)=-1.0*maxClip;
                if (dx(i) > maxClip) dx(i)=maxClip;
            }
            for (int i=0; i<(*(*pgrads)["Wxh"]).size(); i++) {
                if ((*(*pgrads)["Wxh"])(i) < -1.0 * maxClip) (*(*pgrads)["Wxh"])(i)=-1.0*maxClip;
                if ((*(*pgrads)["Wxh"])(i) > maxClip) (*(*pgrads)["Wxh"])(i)=maxClip;
            }
            for (int i=0; i<(*(*pgrads)["Whh"]).size(); i++) {
                if ((*(*pgrads)["Whh"])(i) < -1.0 * maxClip) (*(*pgrads)["Whh"])(i)=-1.0*maxClip;
                if ((*(*pgrads)["Whh"])(i) > maxClip) (*(*pgrads)["Whh"])(i)=maxClip;
            }
            for (int i=0; i<(*(*pgrads)["bh"]).size(); i++) {
                if ((*(*pgrads)["bh"])(i) < -1.0 * maxClip) (*(*pgrads)["bh"])(i)=-1.0*maxClip;
                if ((*(*pgrads)["bh"])(i) > maxClip) (*(*pgrads)["bh"])(i)=maxClip;
            }
            for (int i=0; i<(*(*pgrads)["ho"]).size(); i++) {
                if ((*(*pgrads)["ho"])(i) < -1.0 * maxClip) (*(*pgrads)["ho"])(i)=-1.0*maxClip;
                if ((*(*pgrads)["ho"])(i) > maxClip) (*(*pgrads)["ho"])(i)=maxClip;
            }
        }

        return dx;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* states, t_cppl* pgrads, int id=0) override {
        MatrixN dWxh,dWhh,dbh;
        string name;
        int N=shape(dchain)[0];
        MatrixN dhi,dxi,dphi;
        MatrixN dx(N,T*D);
        dx.setZero();
        for (int t=T-1; t>=0; t--) {
            t_cppl cache{};
            t_cppl grads{};
            t_cppl states{}; // XXX: INIT!
            if (t==T-1) {
                dhi=tensorchunk(dchain,{N,T,H},t);
            } else {
                dhi=tensorchunk(dchain,{N,T,H},t) + dphi;
            }
            // dci <- dchain
            name="t"+std::to_string(t);
            mlPop(name,pcache,&cache);
            dxi=backward_step(dhi,&cache,&states, &grads,id);
            tensorchunkinsert(&dx, dxi, {N,T,D}, t);
            dphi=*grads["ho"]; // XXX cache-rel? -> STATES!
            if (t==T-1) {
                dWxh=*grads["Wxh"];
                dWhh=*grads["Whh"];
                dbh=*grads["bh"];
            } else {
                dWxh+=*grads["Wxh"];
                dWhh+=*grads["Whh"];
                dbh+=*grads["bh"];
            }
            cppl_delete(&grads);
            //cppl_delete(&cache); // XXX uuuhhh
        }
        (*pgrads)["Wxh"] = new MatrixN(dWxh);
        (*pgrads)["Whh"] = new MatrixN(dWhh);
        (*pgrads)["bh"] = new MatrixN(dbh);
        (*pgrads)["ho"] = new MatrixN(dphi);
        // cppl_remove(pcache, "x"); // XXX last cleanup.
        return dx;
    }
};

#endif
