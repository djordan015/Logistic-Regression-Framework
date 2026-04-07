#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <Types.h>


class Optimizer{
    private:
        Gradients calculate_gradients(
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


    public:
        Optimizer(){};

        void apply_step(
            std::vector<double>& weights, 
            double& bias,
            const std::vector<double>& pred,
            const std::vector<std::vector<double>>& X,
            const std::vector<double>& Y, 
            double lr) 
        {
            // get gradients
            Gradients grads = calculate_gradients(pred, X, Y);
            // update weights and bias
            for(size_t i = 0; i < weights.size(); ++i) {
                weights[i] -= lr * grads.dW[i];
            }

            bias -= lr * grads.dB;
        }
};


#endif