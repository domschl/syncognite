#ifndef _CPL_LSTM_H
#define _CPL_LSTM_H

#include "../cp-layers.h"


class LSTM : public Layer {
private:
    int numGpuThreads;
    int numCpuThreads;
    floatN initfactor;
    int T,D,H,N;
    float maxClip=0.0;
    bool nohupdate;
    bool forgetGateInitOnes=true;
    string hname;
    string hname0;
    string cname;
    string cname0;
    void setup(const json& jx) {
        j=jx;
        layerName=j.value("name",(string)"LSTM");
        hname=layerName+"-h";
        hname0=layerName+"-h0";
        cname=layerName+"-c";
        cname0=layerName+"-c0";

        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL | LayerType::LT_EXTERNALSTATE;
        vector<int> inputShape=j.value("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        H=j.value("H",1024);
        N=j.value("N",1);
        XavierMode inittype=xavierInitType(j.value("init",(string)"standard"));
        initfactor=j.value("initfactor",(floatN)1.0);
        maxClip=j.value("clip",(float)0.0);
        forgetGateInitOnes=j.value("forgetgateinitones",true);
        nohupdate=j.value("nohupdate",(bool)false);  // true for auto-diff tests
        D=inputShape[0];
        T=inputShape[1];
        outputShape={H,T};

        cppl_set(&params, "Wxh", new MatrixN(xavierInit(MatrixN(D,4*H),inittype,initfactor)));
        cppl_set(&params, "Whh", new MatrixN(xavierInit(MatrixN(H,4*H),inittype,initfactor)));
        cppl_set(&params, "bh", new MatrixN(xavierInit(MatrixN(1,4*H),inittype,initfactor)));
        if (forgetGateInitOnes)
            params["bh"]->block(0,H,1,H).setOnes();

        //cerr << *params["bh"] << endl;

        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        layerInit=true;
    }
public:
    LSTM(const json& jx) {
        setup(jx);
    }
    LSTM(const string conf) {
        setup(json::parse(conf));
    }
    ~LSTM() {
        cppl_delete(&params);
    }
    virtual void genZeroStates(t_cppl* pstates, int N) override {
        MatrixN *ph= new MatrixN(N,H);
        ph->setZero();
        cppl_set(pstates, hname, ph);
        MatrixN *pc= new MatrixN(N,H);
        pc->setZero();
        cppl_set(pstates, cname, pc);
    }

    MatrixN Sigmoid(const MatrixN& m) {
        MatrixN mn(m);
        MatrixN y;
        // Standard sigmoid is not numerically stable (large -x instable)
        //y=(1.0/(1.0+(mn.array() * -1.0).exp()));
        // Alternative via tanh is stable and doesn't need case distinctions for large + - inf values of x.
        // alt see: http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        y=((mn.array()/2.0).tanh()-1.0)/2.0+1.0;
        return y;
    }
    /*  Forward pass for a single timestep of an LSTM. [CS231]
        The input data has dimension D, the hidden state has dimension H, and we
        use a minibatch size of N.
        Inputs:
        - x: Input data, of shape (N, D)
        - hprev: Previous hidden state, of shape (N, H)
        - cprev: previous cell state, of shape (N, H)
        - Wxh: Input-to-hidden weights, of shape (D, 4H)
        - Whh: Hidden-to-hidden weights, of shape (H, 4H)
        - bh: Biases, of shape (4H,)
        Returns a tuple of:
        - hnext: Next hidden state, of shape (N, H)
        - cnext: Next cell state, of shape (N, H)
        */
    t_cppl forward_step(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) {
        t_cppl cp;
        int N=shape(x)[0];
        MatrixN hprev = *(*pstates)[hname];
        MatrixN cprev = *(*pstates)[cname];
        MatrixN xhx = (hprev * *(params["Whh"]) + x * *(params["Wxh"])).rowwise() + RowVectorN(*params["bh"]);
        MatrixN i = Sigmoid(xhx.block(0,0,N,H));
        MatrixN f = Sigmoid(xhx.block(0,H,N,H));
        MatrixN o = Sigmoid(xhx.block(0,2*H,N,H));
        MatrixN g = (xhx.block(0,3*H,N,H)).array().tanh();
        MatrixN cnext = f.array() * cprev.array() + i.array() * g.array();
        cp[cname0]=new MatrixN(cnext);
        MatrixN tnc=cnext.array().tanh();
        MatrixN hnext=tnc.array() * o.array();
        cp[hname0]=new MatrixN(hnext);
        if (pcache!=nullptr) cppl_set(pcache,"tnc",new MatrixN(tnc));
        if (pcache!=nullptr) cppl_set(pcache,"xhx",new MatrixN(xhx));
        if (pcache!=nullptr) cppl_set(pcache,hname,new MatrixN(hprev));
        if (pcache!=nullptr) cppl_set(pcache,cname,new MatrixN(cprev));
        if (pcache!=nullptr) cppl_set(pcache,"i",new MatrixN(i));
        if (pcache!=nullptr) cppl_set(pcache,"f",new MatrixN(f));
        if (pcache!=nullptr) cppl_set(pcache,"o",new MatrixN(o));
        if (pcache!=nullptr) cppl_set(pcache,"g",new MatrixN(g));
        if (pcache!=nullptr) cppl_set(pcache,"x",new MatrixN(x));
        return cp;
    }

    /*     Backward pass for a single timestep of an LSTM.
        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,) */
    MatrixN backward_step(t_cppl& cp, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) {
        MatrixN hprev = *(*pcache)[hname];
        MatrixN cprev = *(*pcache)[cname];
        MatrixN tnc=*(*pcache)["tnc"];
        MatrixN xhx=*(*pcache)["xhx"];
        MatrixN i=*(*pcache)["i"];
        MatrixN f=*(*pcache)["f"];
        MatrixN o=*(*pcache)["o"];
        MatrixN g=*(*pcache)["g"];
        MatrixN x=*(*pcache)["x"];
        MatrixN dhnext=*cp[hname]; // dhnext
        MatrixN dcnext;
        if (cp.find(cname)!=cp.end()) {
            dcnext=*cp[cname]; // dcnext
        } else {
            dcnext=*cp[hname]; // zero-fake init
            dcnext.setZero(); // Seems to be CS231 api-shortcut of backward() not accomodating dc.
        }

        MatrixN ddo=tnc.array() * dhnext.array();
        MatrixN dtnc=o.array() * dhnext.array();
        dcnext = dcnext +  ((1.0 - tnc.array() * tnc.array()) * dtnc.array()).matrix();
        MatrixN df = cprev.array() * dcnext.array();
        MatrixN dcprev = f.array() * dcnext.array();
        MatrixN di = g.array() * dcnext.array();
        MatrixN dg = i.array() * dcnext.array();
        MatrixN dps(N,4*H);
        MatrixN dps3 = (g.array() * g.array() - 1.0) * -1.0 * dg.array();
        MatrixN dps2 = (o.array() - 1.0) * -1.0 * o.array() * ddo.array();
        MatrixN dps1 = (f.array() -1.0) * -1.0 * f.array() * df.array();
        MatrixN dps0 = (i.array() - 1.0) * -1.0 * i.array() * di.array();
        dps << dps0, dps1, dps2, dps3;  // stack matrices
        MatrixN dphh = dps;
        MatrixN dpshx = dps;
        MatrixN dphx = dpshx.transpose();
        MatrixN dbh = dpshx.colwise().sum();
        MatrixN dx = ((*params["Wxh"]) * dphx).transpose();
        MatrixN dWxh = (dphx * x).transpose();
        MatrixN dhprev = ((*params["Whh"]) * dphh.transpose()).transpose();
        MatrixN dWhh = hprev.transpose() * dphh;

        (*pgrads)["Wxh"] = new MatrixN(dWxh);
        (*pgrads)["Whh"] = new MatrixN(dWhh);
        (*pgrads)["bh"] = new MatrixN(dbh);
        (*pgrads)[hname0] = new MatrixN(dhprev);
        (*pgrads)[cname0] = new MatrixN(dcprev);

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
            for (int i=0; i<(*(*pgrads)[hname0]).size(); i++) {
                if ((*(*pgrads)[hname0])(i) < -1.0 * maxClip) (*(*pgrads)[hname0])(i)=-1.0*maxClip;
                if ((*(*pgrads)[hname0])(i) > maxClip) (*(*pgrads)[hname0])(i)=maxClip;
            }
            for (int i=0; i<(*(*pgrads)[cname0]).size(); i++) {
                if ((*(*pgrads)[cname0])(i) < -1.0 * maxClip) (*(*pgrads)[cname0])(i)=-1.0*maxClip;
                if ((*(*pgrads)[cname0])(i) > maxClip) (*(*pgrads)[cname0])(i)=maxClip;
            }
        }

        return dx;
    }

    MatrixN tensorchunk(const MatrixN& x, vector<int>dims, int b) {
        int A=dims[0];
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
        int C=dims[2];
        int a,c;
        for (a=0; a<A; a++) {
            for (c=0; c<C; c++) {
                (*ph)(a,b*C+c) = ht(a,c);
            }
        }
    }

    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        MatrixN hn(N,T*H);
        MatrixN cn(N,T*H);

        if (x.cols() != D*T) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x:" << shape(x) << " D*T:" << D*T << ", D:" << D << ", T:" << T << ", H:" << H << endl;
            MatrixN h(0,0);
            return h;
        }
        int N=shape(x)[0];

        if (pstates->find(hname)==pstates->end()) {
            MatrixN *ph= new MatrixN(N,H);
            ph->setZero();
            cppl_set(pstates, hname, ph);
        }
        if (pstates->find(cname)==pstates->end()) {
            MatrixN *pc= new MatrixN(N,H);
            pc->setZero();
            cppl_set(pstates, cname, pc);
        }

        MatrixN ht=*(*pstates)[hname];
        MatrixN ct=*(*pstates)[cname];

        for (int t=0; t<T; t++) {
            t_cppl cache{};
            t_cppl states{};
            states[hname]=&ht;
            states[cname]=&ct;
            MatrixN xi=tensorchunk(x,{N,T,D},t);
            t_cppl cp = forward_step(xi,&cache,&states, id);
            ht = *cp[hname0];
            ct = *cp[cname0];
            cppl_delete(&cp);
            tensorchunkinsert(&hn, ht, {N,T,H}, t);
            tensorchunkinsert(&cn, ct, {N,T,H}, t);
            if (pcache!=nullptr) {
                string name="t"+std::to_string(t);
                mlPush(name,&cache,pcache);
            } else cppl_delete(&cache);
        }
        cppl_update(pstates,hname,&ht);
        cppl_update(pstates,cname,&ct);
        if (hn.cols() != (T*H)) {
            cerr << "Inconsistent LSTM-forward result: " << shape(hn) << endl;
        }
        return hn;  // XXX return a t_cppl with cn too, preserver hn, cn as states?
    }

    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        MatrixN dx(N,T*D);

        MatrixN dWxh,dWhh,dbh;
        string name;
        int N=shape(dchain)[0];
        MatrixN dhi,dci,dxi,dphi,dpci;
        dx.setZero();
        for (int t=T-1; t>=0; t--) {
            t_cppl cache{};
            t_cppl grads{};
            t_cppl states{};
            if (t==T-1) {
                dhi=tensorchunk(dchain,{N,T,H},t);
                dci=dhi; // That should also be set by dchain!!
                dci.setZero();
            } else {
                dhi=tensorchunk(dchain,{N,T,H},t) + dphi;
                dci=dpci;
            }
            name="t"+std::to_string(t);
            mlPop(name,pcache,&cache);
            t_cppl cp;
            cp[hname]=new MatrixN(dhi);
            cp[cname]=new MatrixN(dci);
            // cp[cname]=new MatrixN(dci); // XXX: dchain should also take care of dc! -> zero-init for now.
            dxi=backward_step(cp,&cache,&states, &grads,id);
            cppl_delete(&cp);
            tensorchunkinsert(&dx, dxi, {N,T,D}, t);
            dphi=*grads[hname0];
            dpci=*grads[cname0];
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
        }
        (*pgrads)["Wxh"] = new MatrixN(dWxh);
        (*pgrads)["Whh"] = new MatrixN(dWhh);
        (*pgrads)["bh"] = new MatrixN(dbh);
        (*pgrads)[hname0] = new MatrixN(dphi);
        //(*pgrads)[cname0] = new MatrixN(dpci);

        return dx;
    }
};

#endif
