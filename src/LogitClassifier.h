#ifndef LOGITCLASSIFIER_H
#define LOGITCLASSIFIER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <omp.h>

class LogitClassifier {
private:
    double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    double dot_product(const std::vector<double>& features, std::vector<double>& weights){
        double dot = 0;
        for(size_t i = 0; i < features.size(); ++i){
            dot += features[i] * weights[i];
        }

        return dot;
    }

    // Pointer-based overload of dot_product.
    // Takes raw pointers to M features and M weights, returns their dot product.
    double dot_product(const double* features, const double* weights, int M) {
        double dot = 0.0;
        for (int i = 0; i < M; ++i) {
            dot += features[i] * weights[i];
        }
        return dot;
    }

    double predict_probability(const std::vector<double>& features, std::vector<double>& weights, double bias) {
        double z = bias + dot_product(features, weights);
        return sigmoid(z);
    }

    // Pointer-based overload of predict_probability.
    // Computes sigmoid(bias + features . weights) using the pointer-based dot_product.
    double predict_probability(const double* features, const double* weights,
                               int M, double bias) {
        double z = bias + dot_product(features, weights, M);
        return sigmoid(z);
    }

public:
    LogitClassifier(){};

    // forward pass: (X * W) + B then Sigmoid
    std::vector<double> forward_batch(const std::vector<std::vector<double>>& X, std::vector<double>& weights, double bias, bool flag){
        std::vector<double> probs(X.size());

        #pragma omp parallel for schedule(static) if(flag)
        for(int i = 0; i < X.size(); ++i) {
            probs[i] = predict_probability(X[i], weights, bias);
        }

        return probs;
    }

    std::vector<double> forward_batch(const MatrixView& X, const std::vector<double>& weights, double bias, bool flag) {
        std::vector<double> probs(X.N);

        #pragma omp parallel for schedule(static) if(flag)
        for (int i = 0; i < X.N; ++i) {
            const double* row = X.row(i);   // pointer into the flat block
            probs[i] = predict_probability(row, weights.data(), X.M, bias);
        }
        return probs;
    }

    double forward_single(const std::vector<double>& x_sample, std::vector<double>& weights, double bias){
        return predict_probability(x_sample, weights, bias);
    }

    // Pointer-based overload — for SGD when we already have a row pointer from MatrixView
    double forward_single(const double* x_sample, int M, const std::vector<double>& weights, double bias) {
        return predict_probability(x_sample, weights.data(), M, bias);
    }
};

#endif 