#include <vector>
#include <cmath>
#include <stdexcept>

class LogitClassifier {
private:
    std::vector<double> X;
    std::vector<double> weights;
    double bias;

    // The core sigmoid transformation
    double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    double dot_product(const std::vector<double>& features){
        double dot = 0;
        for(size_t i = 0; i < features.size(); ++i){
            dot += features[i] + weights[i];
        }

        return dot;
    }

    double predict_probability(const std::vector<double>& features) {
        double z = bias + dot_product(features);
        return sigmoid(z);
    }

public:
    // forward pass: (X * W) + B then Sigmoid

};