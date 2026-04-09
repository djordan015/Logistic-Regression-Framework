#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include "Optimizer.h"
#include "LogitClassifier.h"
#include "Types.h"

class Model{
private:
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> X_test;
    std::vector<double> Y_train;
    std::vector<double> Y_test;

    std::vector<double> weights;

    double threshold;
    double lr;
    double bias;
    int epochs;
    
    bool print_log;
    LogitClassifier classifier;
    Optimizer optimizer;

    // log loss
    double binary_cross_entropy(std::vector<double> probabilities, std::vector<double>targets){
        double bce = - (1.0/ probabilities.size());
        double loss = 0.0;

        for(int i = 0; i < probabilities.size(); ++i){
            double target = targets[i];
            double p = probabilities[i];
            loss += (target * log(p)) + ((1-target) * log(1-p));
        }
        return bce * loss;
    };


    double learning_rate_schedule(int epoch){
        if(epoch < 10000)
            return 0.1;
        else if(epoch < 20000)
            return 0.01;
        return 0.001;
    }
    

public:
    struct ModelSnapshot {
        std::vector<double> weights;
        double bias;
    };

    Model(const std::vector<std::vector<double>>& X, 
             const std::vector<double>& Y, 
             double th, 
             double initial_lr, 
             int ep) 
    : X_train(X), Y_train(Y), threshold(th), lr(initial_lr), epochs(ep)
    {
        if (X_train.empty() || X_train.size() != Y_train.size()) {
            throw std::invalid_argument("Invalid dataset dimensions."); 
        }
        weights = std::vector<double>(X_train[0].size(), 0.0); 
        bias = 0.0;
        print_log = true;
    }

    const ModelSnapshot get_snapshot() {
        return {weights, bias}; // cite: 2
    }

    void load_snapshot(const std::vector<double>& w, double b) {
        weights = w; // cite: 2
        bias = b;    // cite: 2
    }

    double get_bias(){
        return bias;
    }

    std::vector<double>& get_weights(){
        return weights;
    }


    double get_accuracy(const std::vector<double>& probs, const std::vector<double>& Y_train){
        double accuracy = 0.0;
        int correct = 0;

        for(size_t i = 0; i < probs.size(); ++i) {
            double pred = (probs[i] >= threshold) ? 1.0 : 0.0;
            if(pred == Y_train[i]) correct++;
        }
        
        return static_cast<double>(correct) / probs.size();
    }

    void train(){
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // 1. Get Predictions (Forward Pass)
            std::vector<double> probs = classifier.forward_batch(X_train, weights, bias);

            // 2. Calculate Loss
            double current_loss = binary_cross_entropy(probs, Y_train);

            // 3. Calculate Gradients and update weights
            lr = learning_rate_schedule(epoch);
            optimizer.apply_step(weights, bias, probs, X_train, Y_train, lr);
            
            // 4. Check Accuracy (Optional logging)

            if (print_log && epoch % 100 == 0 ) {
                double entropy = binary_cross_entropy(probs, Y_train);
                double accuracy = get_accuracy(probs, Y_train);

                std::cout << "Epoch: [" << std::setw(5) << epoch << "/" << epochs << "] "
                        << "| Loss: " << std::fixed << std::setprecision(6) << entropy 
                        << "| Acc: " << std::setprecision(2) << (accuracy * 100.0) << "%" 
                        << std::endl;
            }
        }
    }   

    const double test(const std::vector<std::vector<double>>& X_unseen, std::vector<double>& Y_unseen) {
        std::vector<double> probs = classifier.forward_batch(X_unseen, weights, bias);
        double acc = get_accuracy(probs, Y_unseen);
        return acc;
    }
    
};

#endif