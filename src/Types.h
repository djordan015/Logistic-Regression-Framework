#ifndef TYPES_H
#define TYPES_H

#include <vector>

struct Gradients {
    std::vector<double> dW;
    double dB;

    static Gradients calculate_gradients(
            const std::vector<double>& pred, 
            const std::vector<std::vector<double>>& X, 
            const std::vector<double>& Y) 
    {
        int N = X.size(); 
        int num_features = X[0].size();
        
        std::vector<double> dW(num_features, 0.0);
        double dB = 0.0;

        // 1. Iterate over every sample
        for (int i = 0; i < N; ++i) {
            // Calculate error for this specific sample
            double error = pred[i] - Y[i];

            // 2. Accumulate gradient for every weight based on this sample
            for (int j = 0; j < num_features; ++j) {
                dW[j] += error * X[i][j];
            }

            // 3. Accumulate gradient for bias
            dB += error;
        }

        // 4. Average the gradients over the number of samples
        for (int j = 0; j < num_features; ++j) {
            dW[j] /= N;
        }
        dB /= N;

        return {dW, dB};
    }
};

#endif