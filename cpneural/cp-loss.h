#pragma once

#include "cp-neural.h"

/** @brief Categorical cross-entropy loss function.
 *
 * Used to calculate the loss in multi-class classification problems.
 * Inputs are the result of a softmax layer ('probabilities') and the
 * labels of the correct class.
 */
class SparseCategoricalCrossEntropyLoss : public Loss {
  public:
    SparseCategoricalCrossEntropyLoss(const json &jx) {
        /** Sparse categorical cross entropy loss for multi-class classification.
         * @param jx - JSON object with configuration parameters. Not used.
         */
        j=jx;
    }

    virtual floatN loss(MatrixN &yhat, MatrixN &y, t_cppl &pParams) {
        /** Compute the loss for a batch of predictions and targets.
         * @param yhat - predictions (output of softmax layer)
         * @param y - targets, integer class labels i are in y(i,0)
         * @returns loss - cross entropy loss, normalized by the number of samples
         */
        floatN loss = 0.0;
        for (unsigned int i = 0; i < yhat.rows(); i++) {
            if (y(i, 0) >= yhat.cols()) {
                cerr << "internal error: y(" << i << ",0) >= " << yhat.cols()
                     << endl;
                return 1002.0;
            }
            floatN pr_i = yhat(i, (int)y(i, 0));
            if (pr_i == 0.0) {
                cerr << "Invalid zero log-probability at " << i << endl;
                loss += 10000.0;
            } else {
                loss -= log(pr_i);
            }
        }
        loss /= yhat.rows();
        return loss;
    }
};

/** @brief Temporal cross-entropy loss function.
 *
 * Used to calculate the loss in sequence classification problems.
 * Inputs are the result of a softmax layer ('probabilities') and the
 * labels of the correct class.
 */
class TemporalCrossEntropyLoss : public Loss {
  private:
    int T, D;
  public:
    TemporalCrossEntropyLoss(const json &jx) {
        /** Temporal cross entropy loss for sequence classification.
         * @param jx - JSON object with configuration parameters. Should contain vector[int] 'D' and 'T'.
          */
        j = jx;
        vector<int> inputShape = j.value("inputShape", vector<int>{});
        int inputShapeFlat = 1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        D = inputShape[0];
        T = inputShape[1];
    }

    virtual floatN loss(MatrixN &yhat, MatrixN &y, t_cppl *pParams) {
        /** Compute the loss for a batch of predictions and targets.
         * @param yhat - predictions (output of softmax layer)
         * @param y - targets, integer class labels i are in y(i,0)
         * @returns loss - cross entropy loss, normalized by the number of samples
         */
        MatrixN mask;
        int N = (int)y.rows();

        if (pParams->find("mask") == pParams->end()) {
            mask = MatrixN(N, T);
            mask.setOnes();
        } else {
            mask = *((*pParams)["mask"]);
        }

        floatN curLoss = 0.0;
        for (int n = 0; n < N; n++) {
            for (int t = 0; t < T; t++) {
                floatN pr_i = yhat(n * T + t, (int)y(n, t));
                if (pr_i == 0.0) {
                    cerr << "Invalid zero log-probability at n=" << n
                         << "t=" << t << endl;
                    curLoss += 10000.0;
                } else {
                    // cerr << "[" << pr_i << "," << mask(n,t) << "]";
                    curLoss -= log(pr_i) * mask(n, t);
                }
            }
        }
        curLoss /= N;  // Scaling the loss the N*T doesn't work: numerical
                    // differentiation fails.
        return curLoss;
    }
};

/** @brief Mean squared error loss function.
 *
 */
class MeanSquaredErrorLoss : public Loss {
  public:
    MeanSquaredErrorLoss(const json &jx) {
        /** Mean squared error loss function.
         * @param jx - JSON object with configuration parameters. Not used.
         */
        j = jx;
    }

    virtual floatN loss(MatrixN &yhat, MatrixN &y, t_cppl *pParams) {
        /** Compute the loss for a batch of predictions and targets.
         * @param yhat - predictions (e.g. output of softmax layer)
         * @param y - targets, same shape as yhat
         * @returns loss - mean square loss, normalized by the number of samples
         */
        floatN loss = 0.0;
        MatrixN diff = yhat - y;
        for (unsigned int i = 0; i < diff.rows(); i++) {
            for (unsigned int j = 0; j < diff.cols(); j++) {
                loss += diff(i, j) * diff(i, j);
            }
        }
        loss /= yhat.rows();
        return loss;
    }
};

/** @brief Loss Factory: generate a loss class by name */
Loss *lossFactory(string name, const json& j) {
    /** Factory function to create a loss by name.
     * @param name - loss name
     * @param j - JSON object with configuration parameters
     * @returns loss - loss object
     */
    if (name=="SparseCategoricalCrossEntropy") return (Loss *)new SparseCategoricalCrossEntropyLoss(j);
    if (name=="TemporalCrossEntropy") return (Loss *)new TemporalCrossEntropyLoss(j);
    if (name=="MeanSquaredError") return (Loss *)new MeanSquaredErrorLoss(j);
    cerr << "lossFactory called for unknown loss " << name << "." << endl;
    return nullptr;
}

