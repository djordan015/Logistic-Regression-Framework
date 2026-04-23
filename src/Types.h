#ifndef TYPES_H
#define TYPES_H

#include <vector>

struct MatrixView {
    const double* data;
    int N;
    int M;

    // get a pointer to the start of row i
    inline const double* row(int i) const {
        return data + i * M;
    }

    // get the single number at row i, column j
    inline double at(int i, int j) const {
        return data[i * M + j];
    }
};

struct Gradients {
    std::vector<double> dW;
    double dB;

    static Gradients calculate_gradients(
            const std::vector<double>& pred, 
            const std::vector<std::vector<double>>& X, 
            const std::vector<double>& Y,
            const bool& flag) 
    {
        int N = X.size(); 
        int num_features = X[0].size();
        
        std::vector<double> dW(num_features, 0.0);
        double dB = 0.0;

        #pragma omp parallel for reduction(+:dB) reduction(+:dW[:num_features]) schedule(static) if(flag)
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

    //Overloaded function for OMP version that takes MatrixView instead of vector of vectors
    static Gradients calculate_gradients(const std::vector<double>& pred, const MatrixView& X, const std::vector<double>& Y, const bool& flag)
    {
        const int N = X.N;
        const int num_features = X.M;

        std::vector<double> dW(num_features, 0.0);
        double dB = 0.0;

        #pragma omp parallel for reduction(+:dB) reduction(+:dW[:num_features]) schedule(static) if(flag)
        for (int i = 0; i < N; ++i) {
            const double error = pred[i] - Y[i];
            const double* row = X.row(i);

            for (int j = 0; j < num_features; ++j) {
                dW[j] += error * row[j];
            }
            dB += error;
        }

        for (int j = 0; j < num_features; ++j) {
            dW[j] /= N;
        }
        dB /= N;

        return {dW, dB};
    }



    static Gradients calculate_gradients_sgd(
            double pred, 
            const std::vector<double>& x_sample, 
            double y_sample) 
    {
        int num_features = x_sample.size();
        std::vector<double> dW_vec(num_features, 0.0);

        // 1. Calculate error for the single sample
        double error = pred - y_sample;

        // 2. Gradient for weights: dL/dw = (y_hat - y) * x
        for (int j = 0; j < num_features; ++j) {
            dW_vec[j] = error * x_sample[j];
        }

        // 3. Gradient for bias: dL/db = (y_hat - y)
        double dB_val = error;

        // No division by N because N = 1
        return {dW_vec, dB_val};
    }

    static Gradients calculate_gradients_sgd(double pred,
                                         const double* x_sample, int M,
                                         double y_sample) {
        std::vector<double> dW_vec(M, 0.0);
        const double error = pred - y_sample;
        for (int j = 0; j < M; ++j) {
            dW_vec[j] = error * x_sample[j];
        }
        return {dW_vec, error};
    }
    
};

#endif