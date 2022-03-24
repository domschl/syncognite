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

    virtual floatN loss(MatrixN &yhat, MatrixN &y) {
        /** Compute the loss for a batch of predictions and targets.
         * @param yhat - predictions (output of softmax layer
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
            if (pr_i == 0.0)
                cerr << "Invalid zero log-probability at " << i << endl;
            else
                loss -= log(pr_i);
        }
        loss /= yhat.rows();
        return loss;
    }
};
