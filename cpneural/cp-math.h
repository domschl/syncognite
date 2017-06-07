#ifndef _CP_MATH_H
#define _CP_MATH_H

#include <iostream>
#include <vector>

using std::cerr; using std::vector; using std::endl;

enum class MathErr {BAD_DIM, BAD_VAL};

template <typename T>
std::ostream & operator<<(std::ostream & os, vector<T> v) {
    os << "(";
    for (auto i=0; i<(int)v.size(); i++) {
        os << v[i];
        if (i<(int)v.size()-1) os << ", ";
    }
    os << ")";
    return os;
}

template <typename T>
vector<T> operator+(vector<T> a, vector<T> b) {
    if (a.size()!=b.size()) {
        cerr << "Dimensions not compatible vector sum (" << a.size() << ")+("
             << b.size() << ")" << endl;
        throw MathErr::BAD_DIM;
    }
    vector<T> r(a.size());
    for (auto i=0; i<(int)a.size(); i++) {
        r[i] = a[i] + b[i];
    }
    return r;
}
template <typename T>
vector<T> operator-(vector<T> a, vector<T> b) {
    if (a.size()!=b.size()) {
        cerr << "Dimensions not compatible vector diff (" << a.size() << ")+("
             << b.size() << ")" << endl;
        throw MathErr::BAD_DIM;
    }
    vector<T> r(a.size());
    for (auto i=0; i<(int)a.size(); i++) {
        r[i] = a[i] - b[i];
    }
    return r;
}
template <typename T>
vector<T> operator*(vector<T> a, vector<T> b) {
    if (a.size()!=b.size()) {
        cerr << "Dimensions not compatible vector direct product (" << a.size() << ")+("
             << b.size() << ")" << endl;
        throw MathErr::BAD_DIM;
    }
    vector<T> r(a.size());
    for (auto i=0; i<(int)a.size(); i++) {
        r[i] = a[i] * b[i];
    }
    return r;
}
template <typename T>
T operator%(vector<T> a, vector<T> b) {
    if (a.size()!=b.size()) {
        cerr << "Dimensions not compatible dot product (" << a.size() << ")+("
             << b.size() << ")" << endl;
        throw MathErr::BAD_DIM;
    }
    T s=0;
    for (auto i=0; i<(int)a.size(); i++) {
        s += a[i] * b[i];
    }
    return s;
}

template <typename T>
int randomChoice(vector<T> data, vector<T> probabilities) {
    vector<T>accProbs(probabilities);
    if (data.size() != probabilities.size()) {
        cerr  << "Data size " << data.size() << " != probabilities size " << probabilities.size() << endl;
        return -1;
    }
    T cumul=(T)0.0;
    for (int i=0; i<data.size(); i++) {
        cumul += probabilities[i];
        accProbs[i]=cumul;
    }
    T dice=(T)(std::rand()%10000)/10000.0;
    for (int i=0; i<data.size(); i++) {
        if (accProbs[i] > dice) return i;
    }
    cerr << "Your probablity distribution wasn't a probablity (sum dist !=1): " << cumul << endl;
    //cerr << probabilities << endl;
    return (int)data.size()-1;
}

#endif
