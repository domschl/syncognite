#ifndef _CPL_RAN_H
#define _CPL_RAN_H

#include "../cp-layers.h"


class RAN : public Layer {
private:
    int numGpuThreads;
    int numCpuThreads;
    floatN initfactor;
    int T,D,H,N;  // T: time steps, D: input dimension of X, H: number of neurons, N: batch_size
    float maxClip=0.0;
    bool nocupdate;
    string cname;
    string cname0;
    string hname;
    string hname0;
    void setup(const json& jx) {
        j=jx;
        layerName=j.value("name",(string)"RAN");
        layerClassName="RAN";
        cname=layerName+"-c";
        cname0=layerName+"-c0";
        hname=layerName+"-h";
        hname0=layerName+"-h0";

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
        nocupdate=j.value("nocupdate",(bool)false);  // true for auto-diff tests
        D=inputShape[0];
        T=inputShape[1];
        outputShape={H,T};

        cppl_set(&params, "Wxh", new MatrixN(xavierInit(MatrixN(D,3*H),inittype,initfactor)));
        cppl_set(&params, "Whh", new MatrixN(xavierInit(MatrixN(H,2*H),inittype,initfactor)));
        cppl_set(&params, "bh", new MatrixN(xavierInit(MatrixN(1,3*H),inittype,initfactor)));

        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        layerInit=true;
    }
public:
    RAN(const json& jx) {
        setup(jx);
    }
    RAN(const string conf) {
        setup(json::parse(conf));
    }
    ~RAN() {
        cppl_delete(&params);
    }
    
    virtual void genZeroStates(t_cppl* pstates, int N) override {
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
    
    t_cppl forward_step(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) {
        t_cppl cp;
        MatrixN cprev = *(*pstates)[cname];
        MatrixN xhx1 = cprev * *params["Whh"];
        MatrixN xhx2 = (x * *params["Wxh"]).rowwise() + RowVectorN(*params["bh"]);
        MatrixN xhx = xhx1+xhx2.block(0,0,N,2*H);
        MatrixN i=Sigmoid(xhx.block(0,0,N,H));
        MatrixN f=Sigmoid(xhx.block(0,H,N,H));
        MatrixN ctl=xhx2.block(0,2*H,N,H);
        MatrixN cnext=i.array()*ctl.array()+f.array()*cprev.array();
        MatrixN hnext=cnext.array().tanh();
    
        if (pcache != nullptr) {
            (*pcache)["x"] = new MatrixN(x);
            (*pcache)["xhx"] = new MatrixN(xhx);
            (*pcache)["cprev"] = new MatrixN(cprev);
            (*pcache)["cnext"] = new MatrixN(cnext);
            (*pcache)["i"] = new MatrixN(i);
            (*pcache)["f"] = new MatrixN(f);
            (*pcache)["ctl"] = new MatrixN(ctl);
        }
        cp[cname0]=new MatrixN(cnext);
        cp[hname0]=new MatrixN(hnext);
        return cp;
    }

    virtual MatrixN backward_step(t_cppl cp, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) {
        if (pcache->find("x") == pcache->end()) {
            cerr << "cache does not contain x -> fatal!" << endl;
        }
        MatrixN x(*(*pcache)["x"]);
        MatrixN xhx(*(*pcache)["xhx"]);
        MatrixN cprev(*(*pcache)["cprev"]);
        MatrixN cnext(*(*pcache)["cnext"]);
        MatrixN i(*(*pcache)["i"]);
        MatrixN f(*(*pcache)["f"]);
        MatrixN ctl(*(*pcache)["ctl"]);
        MatrixN Wxh(*params["Wxh"]);
        MatrixN Whh(*params["Whh"]);
        MatrixN bh(*params["bh"]);

        MatrixN dhnext=*cp[hname]; // XXX: dhnext
        MatrixN dcnext;
        if (cp.find(cname)!=cp.end()) {
            dcnext=*cp[cname]; // dhnext
        } else {
            dcnext=*cp[hname]; // zero-fake init
            dcnext.setZero(); 
        }

        // MatrixN hnext=cnext.array().tanh();
        MatrixN csq = cnext.array() * cnext.array();
        dcnext = dcnext.array() + (1.0-csq.array()) * dhnext.array();

        // MatrixN cnext=i.array()*ctl.array()+f.array()*cprev.array();
        MatrixN di=ctl.array() * dcnext.array();
        MatrixN dctl=i.array() * dcnext.array();
        MatrixN df=cprev.array() * dcnext.array();
        MatrixN dcprev=f.array() * dcnext.array();

        // MatrixN ctl=xhx2.block(0,2*H,N,H);
        MatrixN dxhx2b=dctl;

        // MatrixN f=Sigmoid(xhx.block(0,H,N,H));
        MatrixN dxhx1b=(1-f.array()*f.array())*df.array();

        // MatrixN i=Sigmoid(xhx.block(0,0,N,H));
        MatrixN dxhx0b=(1-i.array()*i.array())*di.array();

        MatrixN dxhx(N,3*H);
        dxhx << dxhx0b, dxhx1b, dxhx2b;
        
        // MatrixN xhx = xhx1+xhx2.block(0,0,N,2*H);
        MatrixN dxhx1=dxhx;
        MatrixN dxhx2=dxhx.block(0,0,N,2*H);

        // MatrixN xhx2 = (x * *params["Wxh"]).rowwise() + RowVectorN(*params["bh"]);
        // cerr << shape(*params["Wxh"]) << shape(dxhx.transpose()) << shape(x) << endl;
        MatrixN dx=(*params["Wxh"]*dxhx.transpose()).transpose();
        cerr << shape(x) << shape(dxhx) << shape(*params["Wxh"]) << endl;
        MatrixN dWxh=x.transpose()*dxhx;
        MatrixN dbh=dxhx.colwise().sum();

        // MatrixN xhx1 = cprev * *params["Whh"];
        cerr << shape(*params["Whh"]) << shape(dxhx1) << shape(dcnext) << endl;
        MatrixN dc=(*params["Whh"]*dxhx2.transpose()).transpose();
        cerr << shape(cprev) << shape(dxhx1) << shape(*params["Whh"]) << endl;
        MatrixN dWhh=cprev.transpose()*dxhx2;
        
        // MatrixN cprev = *(*pstates)[cname];
        
        /*
        MatrixN hsq = hnext.array() * hnext.array();
        MatrixN hone = MatrixN(hnext);
        hone.setOnes();

        MatrixN t1=(hone-hsq).array() * dchain.array();
        MatrixN t1t=t1.transpose();
        MatrixN dbh=t1.colwise().sum();
        MatrixN dx=(Wxh * t1t).transpose();
        MatrixN dWxh=(t1t * x).transpose();
        MatrixN dh=(Whh * t1t).transpose();
        MatrixN dWhh=hprev.transpose() * t1;
        */
        (*pgrads)["Wxh"] = new MatrixN(dWxh);
        (*pgrads)["Whh"] = new MatrixN(dWhh);
        (*pgrads)["bh"] = new MatrixN(dbh);
        (*pgrads)[cname0] = new MatrixN(dc);

        if (maxClip!=0.0) {
            dx = dx.array().min(maxClip).max(-1*maxClip);
            *(*pgrads)["Wxh"] = (*(*pgrads)["Wxh"]).array().min(maxClip).max(-1*maxClip);
            *(*pgrads)["Whh"] = (*(*pgrads)["Whh"]).array().min(maxClip).max(-1*maxClip);
            *(*pgrads)["bh"] = (*(*pgrads)["bh"]).array().min(maxClip).max(-1*maxClip);
            *(*pgrads)[cname0] = (*(*pgrads)[cname0]).array().min(maxClip).max(-1*maxClip);
        }
        
        return dx;
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

        if (pstates->find(cname)==pstates->end()) {
            MatrixN *pc= new MatrixN(N,H);
            pc->setZero();
            cppl_set(pstates, cname, pc);
        }

        MatrixN ct=*(*pstates)[cname];
        MatrixN ht;

        for (int t=0; t<T; t++) {
            t_cppl cache{};
            t_cppl states{};
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
        cppl_update(pstates,cname,&ct);
        if (hn.cols() != (T*H)) {
            cerr << "Inconsistent RAN-forward result: " << shape(hn) << endl;
        }
        return hn;
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
            cerr << shape(dx) << shape(dxi) << N << "," << T << "," << D << ";" << t << endl;
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
        //(*pgrads)[hname0] = new MatrixN(dphi);
        (*pgrads)[cname0] = new MatrixN(dpci);

        return dx;
    }

};

#endif
