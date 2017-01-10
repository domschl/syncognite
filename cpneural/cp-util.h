#ifndef _CP_UTIL_H
#define _CP_UTIL_H

#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <sstream>

using std::string; using std::map; using std::vector; using std::cerr; using std::endl;

namespace Color {
    enum Code {
        FG_BLACK = 30,
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_YELLOW = 33,
        FG_BLUE     = 34,
        FG_MAGENTA = 35,
        FG_CYAN = 36,
        FG_LIGHT_GRAY = 37,
        FG_DEFAULT  = 39,

        FG_DARK_GRAY = 90,
        FG_LIGHT_RED = 91,
        FG_LIGHT_GREEN = 92,
        FG_LIGHT_YELLOW = 93,
        FG_LIGHT_BLUE = 94,
        FG_LIGHT_MAGENTA = 95,
        FG_LIGHT_CYAN = 96,
        FG_WHITE = 97,

        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };
    class Modifier {
        Code code;
    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream&
        operator<<(std::ostream& os, const Modifier& mod) {
            return os << "\033[" << mod.code << "m";
        }
    };
}

class InputParser{ // http://stackoverflow.com/questions/865668/how-to-parse-command-line-arguments-in-c
    public:
        InputParser (const int &argc, const char **argv){
            for (int i=0; i < argc; ++i) {
                if (i==0) {
                    std::string fpath=argv[0];
                    std::size_t idx = fpath.rfind('/');
                    if (std::string::npos != idx) {
                        std::string filename = fpath.substr(idx + 1);
                        std::string path = fpath.substr(0,idx);
                        this->tokens.push_back("--pogramfilename");
                        this->tokens.push_back(filename);
                        this->tokens.push_back("--pogrampath");
                        this->tokens.push_back(path);
                    }
                } else {
                    this->tokens.push_back(std::string(argv[i]));
                }
            }
        }
        const std::string getOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            return "";
        }
        bool checkOption(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};
/*
int main(int argc, char **argv){
    InputParser input(argc, argv);
    if(input.cmdOptionExists("-h")){
        // Do stuff
    }
    const std::string &filename = input.getCmdOption("-f");
    if (!filename.empty()){
        // Do interesting things ...
    }
    return 0;
}
*/

template <typename T>
using t_param_parser = map<string, T>;

template <typename T>
class ParamParser {
    t_param_parser<int> iparams;
    t_param_parser<vector<int>> viparams;
    t_param_parser<vector<T>> vfparams;
    t_param_parser<T> fparams;
    t_param_parser<bool> bparams;
    t_param_parser<string> sparams;
public:
    ParamParser() {}
    ParamParser(string init) {
        setString(init);
    }
    string trim(const string &s) {
        auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
        //auto wsfront=std::find_if_not(s.begin(),s.end(),std::isspace);
        auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
        //auto wsback=std::find_if_not(s.rbegin(),s.end(),std::isspace).base();
        return (wsback<=wsfront ? string() : string(wsfront,wsback));
    }
    void split(const string &s, char delim, vector<string> &elems) {
        std::stringstream ss;
        ss.str(s);
        string item;
        while (getline(ss, item, delim)) {
            elems.push_back(trim(item));
        }
    }
    string getBlock(string str, string del1, string del2) {
        auto p1=str.find(del1);
        auto p2=str.rfind(del2);
        string r;
        if (p2<=p1) r=string(); else r=str.substr(p1,p2-p1);
        if (r.size()>=del1.size()+del2.size()) r=r.substr(del1.size(),str.size()-del1.size()-del2.size());
        return r;
    }
    void setString(string str) {
        string bl=getBlock(str,"{","}");
        vector<string> tk;
        split(bl,';',tk);
        for (auto t : tk) {
            vector<string> par;
            string p1=getBlock(t,"","=");
            string p2=getBlock(t,"=","");
            if (p1.size()>0) {
                if (p2[0]=='[') { // array
                    string tm=getBlock(p2,"[","]");
                    vector<string> ar;
                    split(tm,',',ar);
                    if (tm.find(".")!=tm.npos || tm.find("e")!=tm.npos) { //float array
                        vector<T> vf;
                        for (auto af : ar) {
                            try {
                                vf.push_back((T)stod(af));
                            } catch (...) {
                                cerr << "EXCEPTION in CpParams 3" << endl;
                            }
                        }
                        setPar(p1,vf);
                    } else { // int array
                        vector<int> vi;
                        for (auto ai : ar) {
                            try {
                                vi.push_back(stoi(ai));
                            } catch (...) {
                                cerr << "EXCEPTION in CpParams 4" << endl;
                            }
                        }
                        setPar(p1,vi);
                    }
                } else if (p2=="true") { // boolean
                    setPar(p1,true);
                } else if(p2=="false") { // boolean
                    setPar(p1,false);
                } else if (p2[0]=='\'') { //string
                    // XXX: (de/)encode escape stuff:   ;{}
                    string st=getBlock(p2,"'", "'");
                    setPar(p1,st);
                } else if (p2.find(".")!=p2.npos || p2.find("e")!=p2.npos) { //float
                    try {
                        setPar(p1,stof(p2));
                    } catch (...) {
                        cerr << "EXCEPTION in CpParams 1" << endl;
                    }
                } else { //assume int
                    try {
                        setPar(p1,stoi(p2));
                    } catch (...) {
                        cerr << "EXCEPTION in CpParams 2" << endl;
                    }
                }
            }
        }
    }
    string getString(bool pretty=true) {
        string ind,sep,asep, ter, tnl;
        string qt=""; // or: "\"";

        if (pretty) {
            ind="  "; sep="="; asep=", "; ter=";\n"; tnl="\n";
        } else {
            ind=""; sep="="; asep=","; ter=";"; tnl="";
        }
        string sdl="'";
        string str="{"+tnl;
        for (auto it : fparams) str += ind + qt+it.first + qt+sep + std::to_string(it.second) + ter;
        for (auto it : iparams) str += ind + qt+it.first + qt+sep + std::to_string(it.second) + ter;
        for (auto it : viparams) {
            str += ind + qt+it.first + qt+sep + "[";
            bool is=false;
            for (auto i : it.second) {
                if (is) str+=asep;
                is=true;
                str+=std::to_string(i);
            }
            str+="]"+ter;
        }
        for (auto it : vfparams) {
            str += ind + qt+it.first + qt+sep + "[";
            bool is=false;
            for (auto f : it.second) {
                if (is) str+=asep;
                is=true;
                str+=std::to_string(f);
            }
            str+="]"+ter;
        }
        for (auto it : bparams) {
            str += ind + qt+it.first + qt+sep;
            if (it.second) str+="true";
            else str+="false";
            str+=ter;
        }
        for (auto it : sparams) {
            // XXX: decode escape stuff:   ;{}
            str += ind + qt+it.first + qt+sep +sdl+it.second +sdl+ ter;
        }
        str+="}"+tnl;
        return str;
    }
    bool isDefined(string par) {
        if (iparams.find(par)!=iparams.end()) return true;
        if (viparams.find(par)!=viparams.end()) return true;
        if (vfparams.find(par)!=vfparams.end()) return true;
        if (fparams.find(par)!=fparams.end()) return true;
        if (bparams.find(par)!=bparams.end()) return true;
        // XXX Warning: Strings lake even rudimentary escape sequence handling!
        if (sparams.find(par)!=sparams.end()) return true;
        return false;
    }
    void nerase(string par) {
        if (fparams.find(par)!=fparams.end()) fparams.erase(par);
        if (iparams.find(par)!=iparams.end()) iparams.erase(par);
        if (viparams.find(par)!=viparams.end()) viparams.erase(par);
        if (vfparams.find(par)!=vfparams.end()) vfparams.erase(par);
        if (bparams.find(par)!=bparams.end()) bparams.erase(par);
        // XXX Warning: Strings lake even rudimentary escape sequence handling!
        if (sparams.find(par)!=sparams.end()) sparams.erase(par);
    }
    T getPar(string par, T def=0.0) {
        auto it=fparams.find(par);
        if (it==fparams.end()) {
            fparams[par]=def;
        }
        return fparams[par];
    }
    int getPar(string par, int def=0) {
        auto it=iparams.find(par);
        if (it==iparams.end()) {
            iparams[par]=def;
        }
        return iparams[par];
    }
    vector<int> getPar(string par, vector<int> def={}) {
        auto it=viparams.find(par);
        if (it==viparams.end()) {
            viparams[par]=def;
        }
        return viparams[par];
    }
    vector<T> getPar(string par, vector<T> def={}) {
        auto it=vfparams.find(par);
        if (it==vfparams.end()) {
            vfparams[par]=def;
        }
        return vfparams[par];
    }
    bool getPar(string par, bool def=false) {
        auto it=bparams.find(par);
        if (it==bparams.end()) {
            bparams[par]=def;
        }
        return bparams[par];
    }
    // Traveler beware: cast argument to (string), if using string-constants, otherwise it's taken for int-array!
    string getPar(string par, string def="") {
        auto it=sparams.find(par);
        if (it==sparams.end()) {
            // XXX: decode escape stuff:   ;{}
            sparams[par]=def;
        }
        return sparams[par];
    }
    void setPar(string par, int val) {
        nerase(par);
        iparams[par]=val;
    }
    void setPar(string par, vector<int> val) {
        nerase(par);
        viparams[par]=val;
    }
    void setPar(string par, vector<T> val) {
        nerase(par);
        vfparams[par]=val;
    }
    void setPar(string par, T val) {
        nerase(par);
        fparams[par]=val;
    }
    void setPar(string par, bool val) {
        nerase(par);
        bparams[par]=val;
    }
    // Traveler beware: cast argument to (string), if using string-constants, otherwise it's taken for int-array!
    void setPar(string par, string val) {
        nerase(par);
        // XXX: encode escape stuff:   ;{}
        sparams[par]=val;
    }
};

#endif
