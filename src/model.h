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
    // const std::vector<std::vector<double>>& X_train;
    // const std::vector<std::vector<double>>& X_test;
    // const std::vector<double>& Y_train;
    // const std::vector<double>& Y_test;

    std::vector<double> weights;
    double threshold;
    double bias;
    int epochs;
    
    bool print_log;
    LogitClassifier classifier;

    // Ensure X and Y are valid and compatible
    void validate_data(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) const {
        if (X.empty()) {
            throw std::invalid_argument("Dataset is empty.");
        }
        if (X.size() != Y.size()) {
            throw std::invalid_argument("Features and labels must have the same number of samples.");
        }
    }

    // Weight Initialization
    void initialize_parameters(int num_features) {
        if (weights.empty()) { // Only initialize if not already trained
            weights = std::vector<double>(num_features, 0.0);
            bias = 0.0;
        }
    }

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

    Model( double th, int ep, bool debug = false) 
    : threshold(th), epochs(ep), print_log(debug) {
        classifier = LogitClassifier();
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

    void set_epochs(int ep){
        epochs = ep;
    }

    void set_threshold(double th){
        threshold = th;
    }

    void train(const std::vector<std::vector<double>>& X_train, 
            const std::vector<double> Y_train, 
            Optimizer& opt) {
                
                validate_data(X_train, Y_train);
                initialize_parameters(X_train[0].size());
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // 1. Get Predictions (Forward Pass)
            std::vector<double> probs = classifier.forward_batch(X_train, weights, bias);


            // 2. Calculate Gradients 
            Gradients grads = Gradients::calculate_gradients(probs, X_train, Y_train);

            // 3. Update Weights (weights updated by optimizer)
            double lr = learning_rate_schedule(epoch);
            opt.apply_step(weights, bias, grads, lr);
            
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