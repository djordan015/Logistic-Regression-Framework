#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <random>  
#include <algorithm> 
#include <numeric>   
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
    double learning_rate;
    
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
    double binary_cross_entropy(std::vector<double> probabilities, std::vector<double> targets) {
        double bce = - (1.0 / probabilities.size());
        double loss = 0.0;
        const double eps = 1e-15; // Tiny value to prevent log(0)

        for(int i = 0; i < probabilities.size(); ++i) {
            double target = targets[i];
            double p = probabilities[i];
            
            // Clip p to be in range [eps, 1 - eps]
            if (p < eps) p = eps;
            if (p > 1.0 - eps) p = 1.0 - eps;

            loss += (target * log(p)) + ((1 - target) * log(1 - p));
        }
        return bce * loss;
    }


    double learning_rate_schedule(int epoch, int N) {
        // Base LR: smaller for massive datasets to prevent explosion
        double base_lr = (N > 100000) ? 0.0001 : 0.01;
        
        // Decay factor: determines how fast the LR shrinks
        // Increasing this makes the LR drop faster
        double decay = 0.01; 

        // Formula: lr = base_lr / (1 + decay * epoch)
        return base_lr / (1.0 + decay * epoch);
    }

    void print_results(const std::vector<std::vector<double>>& X_train, 
        const std::vector<double>& Y_train,
        const int epoch){

        std::vector<double> current_probs = classifier.forward_batch(X_train, weights, bias);
        double entropy = binary_cross_entropy(current_probs, Y_train);
        double accuracy = get_accuracy(current_probs, Y_train);

        std::cout << "Epoch: [" << std::setw(5) << epoch << "/" << epochs << "] "
                << "| Loss: " << std::fixed << std::setprecision(6) << entropy 
                << "| Acc: " << std::setprecision(2) << (accuracy * 100.0) << "%" 
                << std::endl;
            
    }
        

public:
    struct ModelSnapshot {
        std::vector<double> weights;
        double bias;
    };

    Model(double lr, double th, int ep, bool debug = false) 
    : learning_rate(lr), threshold(th), epochs(ep), print_log(debug){
        classifier = LogitClassifier();
    }

    const ModelSnapshot get_snapshot() {
        return {weights, bias}; 
    }

    void load_snapshot(const std::vector<double>& w, double b) {
        weights = w; 
        bias = b;    
    }

    double get_bias(){
        return bias;
    }

    std::vector<double>& get_weights(){
        return weights;
    }

    double get_accuracy(const std::vector<double>& probs, const std::vector<double>& Y){
        double accuracy = 0.0;
        int correct = 0;

        for(size_t i = 0; i < probs.size(); ++i) {
            double pred = (probs[i] >= threshold) ? 1.0 : 0.0;
            if(pred == Y[i]) correct++;
        }
        
        return static_cast<double>(correct) / probs.size();
    }

    void set_epochs(int ep){
        epochs = ep;
    }

    void set_threshold(double th){
        threshold = th;
    }

    void set_learing_rate(double lr){
        learning_rate = lr;
    }
    void train(const std::vector<std::vector<double>>& X_train, 
            const std::vector<double>& Y_train, 
            Optimizer& opt,
            bool use_omp) {

        if (use_omp){
            // std::cout << "TRAINING WITH OMP" << std::endl;
            train_omp(X_train, Y_train, opt);
        }
        else{
            train_og(X_train, Y_train, opt);
        }
    }



    void train_omp(const std::vector<std::vector<double>>& X_train, 
            const std::vector<double>& Y_train, 
            Optimizer& opt) {

        validate_data(X_train, Y_train);
        initialize_parameters(X_train[0].size());
        
        const int interval = static_cast<int>(epochs * 0.1);
        const int N = X_train.size();
        bool is_sgd = false;
        if (SGD* sgd_ptr = dynamic_cast<SGD*>(&opt)) {
            is_sgd = true;
        }
        

        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());

        for (int epoch = 0; epoch <= epochs; ++epoch) {
            
            if(is_sgd){
                std::shuffle(indices.begin(), indices.end(), g);
                for (int step = 0; step < N; ++step) {
                    int i = indices[step];

                    double prob = classifier.forward_single(X_train[i], weights, bias);
                    Gradients grads = Gradients::calculate_gradients_sgd(prob, X_train[i], Y_train[i]);

                    // double lr = 0.1;
                    opt.apply_step_omp(weights, bias, grads, learning_rate);
                    }
            }
            else{
                // Batch Gradient Descent
                std::vector<double> probs = classifier.forward_batch(X_train, weights, bias, true);
                Gradients grads = Gradients::calculate_gradients(probs, X_train, Y_train, true);

                // double lr = 0.1;
                opt.apply_step_omp(weights, bias, grads, learning_rate);
            }
            
            // Check Accuracy
            if ((print_log && epoch % interval == 0 ) || (print_log && epoch == epochs)) {
                print_results(X_train, Y_train, epoch);
            }
        }
    }
    
    
    void train_og(const std::vector<std::vector<double>>& X_train, 
            const std::vector<double>& Y_train, 
            Optimizer& opt) {
                
        validate_data(X_train, Y_train);
        initialize_parameters(X_train[0].size());
        
        const int interval = static_cast<int>(epochs * 0.1);
        const int N = X_train.size();
        bool is_sgd = false;
        if (SGD* sgd_ptr = dynamic_cast<SGD*>(&opt)) {
            is_sgd = true;
        }
        

        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());

        for (int epoch = 0; epoch <= epochs; ++epoch) {
            
            if(is_sgd){
                std::shuffle(indices.begin(), indices.end(), g);
                for (int step = 0; step < N; ++step) {
                    int i = indices[step];

                    double prob = classifier.forward_single(X_train[i], weights, bias);
                    Gradients grads = Gradients::calculate_gradients_sgd(prob, X_train[i], Y_train[i]);

                    // double lr = 0.1;
                    opt.apply_step(weights, bias, grads, learning_rate);
                    }
            }
            else{
                // Batch Gradient Descent
                std::vector<double> probs = classifier.forward_batch(X_train, weights, bias, false);
                Gradients grads = Gradients::calculate_gradients(probs, X_train, Y_train, false);

                // double lr = 0.1;
                opt.apply_step(weights, bias, grads, learning_rate);
            }
            
            // Check Accuracy
            if ((print_log && epoch % interval == 0 ) || (print_log && epoch == epochs)) {
                print_results(X_train, Y_train, epoch);
            }
        }
    }   

    const double test(const std::vector<std::vector<double>>& X_unseen, const std::vector<double>& Y_unseen) {
        std::vector<double> probs = classifier.forward_batch(X_unseen, weights, bias);
        double acc = get_accuracy(probs, Y_unseen);
        return acc;
    }
    
};

#endif
