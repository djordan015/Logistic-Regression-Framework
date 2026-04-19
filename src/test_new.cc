#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <iomanip>
#include "model.h"
#include "Optimizer.h"

/**
 * Generates a synthetic dataset for binary classification.
 * Rule: If sum(features) > 0, Label = 1.0, else 0.0.
 */
void generate_synthetic_data(int samples, int features, 
                             std::vector<std::vector<double>>& X, 
                             std::vector<double>& Y) {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    X.reserve(samples);
    Y.reserve(samples);

    for (int i = 0; i < samples; ++i) {
        std::vector<double> row;
        double sum = 0.0;
        for (int j = 0; j < features; ++j) {
            double val = dist(gen);
            row.push_back(val);
            sum += val;
        }
        X.push_back(row);
        Y.push_back(sum > 0 ? 1.0 : 0.0);
    }
}

void test_optimizer(const std::string& name, 
                    Optimizer& opt, 
                    const std::vector<std::vector<double>>& X_train, 
                    const std::vector<double>& Y_train,
                    const std::vector<std::vector<double>>& X_test, 
                    const std::vector<double>& Y_test,
                    int epochs, 
                    double lr,
                    double th) {

    std::cout << "\n>>> Testing Optimizer: " << name << " <<<" << std::endl;

    // Initialize Model
    // Note: We use a fresh model for each optimizer to ensure a fair test
    Model model(lr, th, epochs, true); 

    auto start = std::chrono::high_resolution_clock::now();
    
    model.train(X_train, Y_train, opt);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    double final_acc = model.test(X_test, Y_test);

    std::cout << "\n[" << name << "] Results:" << std::endl;
    std::cout << "Training Time: " << std::fixed << std::setprecision(4) << diff.count() << " seconds" << std::endl;
    std::cout << "Final Test Accuracy: " << (final_acc * 100.0) << "%" << std::endl;

    assert(final_acc > 0.80); 
    std::cout << "SUCCESS: " << name << " passed accuracy requirements." << std::endl;
}

int main() {
    const int NUM_FEATURES = 10;
    const int TRAIN_SAMPLES = 5000; // Reduced slightly for faster feedback during dev
    const int TEST_SAMPLES = 500;
    const int EPOCHS = 500;
    const double LEARNING_RATE = 0.1;
    const double THRESHOLD = 0.65;

    try {
        std::cout << "--- Starting Comparative Test Suite ---" << std::endl;
        
        // 1. Prepare Data
        std::vector<std::vector<double>> X_train, X_test;
        std::vector<double> Y_train, Y_test;
        generate_synthetic_data(TRAIN_SAMPLES, NUM_FEATURES, X_train, Y_train);
        generate_synthetic_data(TEST_SAMPLES, NUM_FEATURES, X_test, Y_test);

        // 2. Test Batch Gradient Descent
        GradientDescent gd_opt;
        test_optimizer("Batch Gradient Descent", gd_opt, X_train, Y_train, X_test, Y_test, EPOCHS, LEARNING_RATE, THRESHOLD);

        std::cout << "\n--------------------------------------------" << std::endl;

        // 3. Test Stochastic Gradient Descent
        SGD sgd_opt;
        test_optimizer("Stochastic Gradient Descent", sgd_opt, X_train, Y_train, X_test, Y_test, EPOCHS, LEARNING_RATE, THRESHOLD);

    } catch (const std::exception& e) {
        std::cerr << "Test suite failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}