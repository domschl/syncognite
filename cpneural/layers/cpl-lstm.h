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
    string hname;
    string hname0;
    string cname;
    string cname0;
    void setup(const CpParams& cx) {
        cp=cx;
        layerName=cp.getPar("name",(string)"LSTM");
        hname=layerName+"-h";
        hname0=layerName+"-h0";
        cname=layerName+"-c";
        cname0=layerName+"-c0";

        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL | LayerType::LT_EXTERNALSTATE;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        H=cp.getPar("H",1024);
        N=cp.getPar("N",1);
        XavierMode inittype=xavierInitType(cp.getPar("init",(string)"standard"));
        initfactor=cp.getPar("initfactor",(floatN)1.0);
        maxClip=cp.getPar("clip",(float)0.0);
        nohupdate=cp.getPar("nohupdate",(bool)false);  // true for auto-diff tests
        D=inputShape[0];
        T=inputShape[1];
        outputShape={H,T};

        cppl_set(&params, "Wxh", new MatrixN(xavierInit(MatrixN(D,4*H),inittype,initfactor)));
        cppl_set(&params, "Whh", new MatrixN(xavierInit(MatrixN(H,4*H),inittype,initfactor)));
        cppl_set(&params, "bh", new MatrixN(xavierInit(MatrixN(1,4*H),inittype,initfactor)));
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        layerInit=true;
    }
public:
    LSTM(const CpParams& cx) {
        setup(cx);
    }
    LSTM(const string conf) {
        setup(CpParams(conf));
    }
    ~LSTM() {
        cppl_delete(&params);
    }
    virtual void genZeroStates(t_cppl* pstates, int N) override {
        MatrixN *ph= new MatrixN(N,H);
        ph->setZero();
        cppl_set(pstates, hname, ph);
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
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
        next_h, next_c, Wx, Wh, prev_h, prev_c, x, tnc, o, f, g, i, ps = cache
        H = prev_h.shape[1]

        # next_h = o * tnc
        do = tnc * dnext_h
        dtnc = o * dnext_h
        # tnc = np.tanh(next_c)
        dnext_c += (1 - tnc * tnc) * dtnc
        # next_c = fpc + ig
        dfpc = dnext_c
        dig = dnext_c
        # fpc = f * prev_c
        df = prev_c * dfpc
        dprev_c = f * dfpc
        # ig = i * g
        di = g * dig
        dg = i * dig
        # g = np.tanh(ps[:,3*H:4*H])
        dps = np.zeros(ps.shape)
        dps[:, 3*H:4*H] = (np.ones(prev_h.shape) - g ** 2) * dg
        # o = sigmoid(ps[:,2*H:3*H])
        dps[:, 2*H:3*H] = (np.ones(prev_h.shape) - o) * o * do
        # f = sigmoid(ps[:,H:2*H])
        dps[:, H:2*H] = (np.ones(prev_h.shape) - f) * f * df
        # i = sigmoid(ps[:,:H])
        dps[:, :H] = (np.ones(prev_h.shape) - i) * i * di
        # ps = phh + pshx
        dphh = dps
        dpshx = dps
        # pshx = phx + b
        dphx = dpshx.T
        db = np.sum(dpshx, axis=0)
        # phx = np.dot(x, Wx)
        dx = np.dot(Wx, dphx).T
        dWx = np.dot(dphx, x).T
        # phh = np.dot(prev_h, Wh)
        dprev_h = np.dot(Wh, dphh.T).T
        dWh = np.dot(prev_h.T, dphh)
        return dx, dprev_h, dprev_c, dWx, dWh, db
    */
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
        MatrixN dcnext=*cp[cname]; // dcnext

        // # next_h = o * tnc
        // do = tnc * dnext_h
        // dtnc = o * dnext_h

        MatrixN ddo=tnc.array() * dhnext.array();
        MatrixN dtnc=o.array() * dhnext.array();
        // # tnc = np.tanh(next_c)
        // dnext_c += (1 - tnc * tnc) * dtnc
        dcnext = dcnext +  ((1.0 - tnc.array() * tnc.array()) * dtnc.array()).matrix();
        // dfpc = dnext_c
        // dig = dnext_c
        // fpc = f * prev_c
        MatrixN df = cprev.array() * dcnext.array();
        // dprev_c = f * dfpc
        MatrixN dcprev = f.array() * dcnext.array();

        cerr << "dhnext: " << shape(dhnext) << dhnext << endl;
        cerr << "o: " << shape(o) << o << endl;
        cerr << "dtnc: " << shape(dtnc) << dtnc << endl;
        cerr << "tnc: " << shape(tnc) << tnc << endl;
        cerr << "dcnext: " << shape(dcnext) << dcnext << endl;
        cerr << "dcnext+: " << shape(dcnext) << dcnext << endl;
        cerr << "f: " << shape(f) << f << endl;
        cerr << "dcprev:" << shape(dcprev) << dcprev << endl;


        // ig = i * g
        MatrixN di = g.array() * dcnext.array();
        MatrixN dg = i.array() * dcnext.array();


        //# g = np.tanh(ps[:,3*H:4*H])
        //dps = np.zeros(ps.shape)
        MatrixN dps(N,4*H);
        //dps[:, 3*H:4*H] = (np.ones(prev_h.shape) - g ** 2) * dg
        MatrixN dps3 = (g.array() * g.array() - 1.0) * -1.0 * dg.array();
        // o = sigmoid(ps[:,2*H:3*H])
        // dps[:, 2*H:3*H] = (np.ones(prev_h.shape) - o) * o * do
        MatrixN dps2 = (o.array() - 1.0) * -1.0 * o.array() * ddo.array();
        // f = sigmoid(ps[:,H:2*H])
        //dps[:, H:2*H] = (np.ones(prev_h.shape) - f) * f * df
        MatrixN dps1 = (f.array() -1.0) * -1.0 * f.array() * df.array();
        //# i = sigmoid(ps[:,:H])
        //dps[:, :H] = (np.ones(prev_h.shape) - i) * i * di
        MatrixN dps0 = (i.array() - 1.0) * -1.0 * i.array() * di.array();
        //# ps = phh + pshx
        dps << dps0, dps1, dps2, dps3;  // stack matrices
        MatrixN dphh = dps;
        MatrixN dpshx = dps;
        // pshx = phx + b
        MatrixN dphx = dpshx.transpose();
        MatrixN dbh = dpshx.colwise().sum();  // np.sum(dpshx, axis=0)
        // phx = np.dot(x, Wx)
        MatrixN dx = ((*params["Wxh"]) * dphx).transpose(); // np.dot(Wx, dphx).T
        MatrixN dWxh = (dphx * x).transpose();
        // phh = np.dot(prev_h, Wh)
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
        /*
        if (x.cols() != D*T) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x:" << shape(x) << " D*T:" << D*T << ", D:" << D << ", T:" << T << ", H:" << H << endl;
            MatrixN h(0,0);
            return h;
        }
        int N=shape(x)[0];

        if (pstates->find(hname)==pstates->end()) {
            genZeroStates(pstates, N);  // If states[hname] is not defined, initialize to zero!
        }

        MatrixN ht=*(*pstates)[hname];

        for (int t=0; t<T; t++) {
            t_cppl cache{};
            t_cppl states{};
            states[hname]=&ht;
            MatrixN xi=tensorchunk(x,{N,T,D},t);
            ht = forward_step(xi,&cache,&states, id);
            tensorchunkinsert(&hn, ht, {N,T,H}, t);
            if (pcache!=nullptr) {
                string name="t"+std::to_string(t);
                mlPush(name,&cache,pcache);
            } else cppl_delete(&cache);
        }
        cppl_update(pstates,hname,&ht);
        if (hn.cols() != (T*H)) {
            cerr << "Inconsistent LSTM-forward result: " << shape(hn) << endl;
        }
        */
        return hn;
    }

    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        MatrixN dx(N,T*D);
        /*
        MatrixN dWxh,dWhh,dbh;
        string name;
        int N=shape(dchain)[0];
        MatrixN dhi,dxi,dphi;
        dx.setZero();
        for (int t=T-1; t>=0; t--) {
            t_cppl cache{};
            t_cppl grads{};
            t_cppl states{};
            if (t==T-1) {
                dhi=tensorchunk(dchain,{N,T,H},t);
            } else {
                dhi=tensorchunk(dchain,{N,T,H},t) + dphi;
            }
            name="t"+std::to_string(t);
            mlPop(name,pcache,&cache);
            dxi=backward_step(dhi,&cache,&states, &grads,id);
            tensorchunkinsert(&dx, dxi, {N,T,D}, t);
            dphi=*grads[hname0];
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
        */
        return dx;
    }
};

#endif
