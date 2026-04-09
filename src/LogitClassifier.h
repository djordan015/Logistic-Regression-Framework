#ifndef LOGITCLASSIFIER_H
#define LOGITCLASSIFIER_H

#include <vector>
#include <cmath>
#include <stdexcept>

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

    double predict_probability(const std::vector<double>& features, std::vector<double>& weights, double bias) {
        double z = bias + dot_product(features, weights);
        return sigmoid(z);
    }

public:
    LogitClassifier(){};

    // forward pass: (X * W) + B then Sigmoid
    std::vector<double> forward_batch(const std::vector<std::vector<double>>& X, std::vector<double>& weights, double bias){
        std::vector<double> probs(X.size());
        for(int i = 0; i < X.size(); ++i) {
            probs[i] = predict_probability(X[i], weights, bias);
        }

        return probs;
    }
};

#endif