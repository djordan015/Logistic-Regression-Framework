#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <Types.h>

class Optimizer {
public:
    virtual ~Optimizer() {}

    // The Model calls this to update its state
    virtual void apply_step(
        std::vector<double>& weights, 
        double& bias,
        const Gradients& grads,
        double lr) = 0; // Pure virtual function
    
    virtual void apply_step_omp(
        std::vector<double>& weights, 
        double& bias,
        const Gradients& grads,
        double lr) = 0;
    
};

class SGD : public Optimizer {
public:
    void apply_step(
        std::vector<double>& weights, 
        double& bias,
        const Gradients& grads,
        double lr)  
    {
        for(size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= lr * grads.dW[i];
        }
        bias -= lr * grads.dB;
    }

    void apply_step_omp(
        std::vector<double>& weights, 
        double& bias,
        const Gradients& grads,
        double lr)  
    {   
        #pragma parallel for
        for(size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= lr * grads.dW[i];
        }
        bias -= lr * grads.dB;
    }
};

class GradientDescent : public Optimizer {
public:

    GradientDescent() = default;

    void apply_step(
        std::vector<double>& weights, 
        double& bias,
        const Gradients& grads,
        double lr)  
    {
        // 1. Update each weight in the feature vector
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= lr * grads.dW[i];
        }

        // 2. Update the bias term
        bias -= lr * grads.dB;
    }

    void apply_step_omp(
        std::vector<double>& weights, 
        double& bias,
        const Gradients& grads,
        double lr)  
    {   
        #pragma parallel for
        for(size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= lr * grads.dW[i];
        }
        bias -= lr * grads.dB;
    }
};



#endif