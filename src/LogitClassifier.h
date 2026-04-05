#ifndef LOGITCLASSIFIER_H
#define LOGITCLASSIFIER_H


#include <vector>
#include <cmath>
#include <stdexcept>

// logit classifier for a SINGLE input x
class LogitClassifier {
private:
    std::vector<double> weights;
    double bias;

    // The core sigmoid transformation
    double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    double dot_product(const std::vector<double>& features){
        double dot = 0;
        for(size_t i = 0; i < features.size(); ++i){
            dot += features[i] * weights[i];
        }

        return dot;
    }

    double predict_probability(const std::vector<double>& features) {
        double z = bias + dot_product(features);
        return sigmoid(z);
    }

public:
    LogitClassifier(int num_features) : weights(num_features, 0.0), bias(0.0) {}

    // forward pass: (X * W) + B then Sigmoid
    std::vector<double> forward_batch(const std::vector<std::vector<double>>& X) {
        std::vector<double> predictions(X.size());

        for(int i = 0; i < X.size(); ++i) {
            predictions[i](predict_probability(sample));
        }
        return predictions;
    }

    std::vector<double>& get_weights() { return weights; }
    double& get_bias() { return bias; }

};

#endif