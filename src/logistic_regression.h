#include <vector>
#include <cmath>
#include <stdexcept>

class LogitClassifier {
private:
    std::vector<double> weights;
    double bias;

    // The core sigmoid transformation
    double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

public:
    // forward pass: (X * W) + B then Sigmoid
    double predict_probability(const std::vector<double>& features) {
        double z = bias;
        for (size_t i = 0; i < features.size(); ++i) {
            z += features[i] * weights[i];
        }
        return sigmoid(z);
    }
};