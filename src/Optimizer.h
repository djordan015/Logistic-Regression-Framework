#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <Types.h>


class Optimizer{
    private:
        double learning_rate;

        std::vector<double> calculate_gradients(
            const std::vector<double> probs,
            const std::vector<std::vector<double>>& X,
            const std::vector<double>& Y) 
        {
            int N = X.size(); // num samples
            int num_features = X[0].size(); // num weights
            std::vector<double> dW(num_features, 0.0);
            double dB = 0.0;

            for(int i = 0; i < num_features; ++i){
                double sum_error = 0.0;
                double error = probs[i] - Y[i];


                // calcualte error for each feature
                for(int j = 0; j < N; ++j){
                    sum_error += error * X[i][j];
                }

                // average gradeint for weight_i
                dW[i] = sum_error / N;
                dB += error;
            }
            dB /= N;
            return dW;
        }


    public:
        Optimizer(double lr) : learning_rate(lr){}

        void apply_step(
            std::vector<double>& weights, 
            double& bias,
            const std::vector<std::vector<double>>& X,
            const std::vector<double>& Y, 
            const std::vector<double>& probs,
            double lr) 
        {
            // get gradients
            std::vector<double> gradients = calculate_gradients(probs, X, Y);
            for(size_t i = 0; i < weights.size(); ++i) {
                weights[i] -= lr * gradients[i];
            }
            
            // update bias
        }
};


#endif