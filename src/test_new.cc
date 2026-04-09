#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include "model.h"
#include "Optimizer.h"

/**
 * Generates a synthetic dataset for binary classification.
 * Rule: If sum(features) > 0, Label = 1.0, else 0.0.
 */
void generate_synthetic_data(int samples, int features, 
                             std::vector<std::vector<double>>& X, 
                             std::vector<double>& Y) {
    // Fixed seed for reproducibility
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
        // Label based on simple linear boundary
        Y.push_back(sum > 0 ? 1.0 : 0.0);
    }
}

void run_large_test() {
    const int NUM_FEATURES = 10;
    const int TRAIN_SAMPLES = 20000;
    const int TEST_SAMPLES = 1000;
    const int EPOCHS = 500;

    std::cout << "--- Starting Large Scale Test Suite ---" << std::endl;
    std::cout << "Config: " << TRAIN_SAMPLES << " samples, " << NUM_FEATURES << " features." << std::endl;

    // 1. Generate Training and Testing Data
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> Y_train, Y_test;

    generate_synthetic_data(TRAIN_SAMPLES, NUM_FEATURES, X_train, Y_train);
    generate_synthetic_data(TEST_SAMPLES, NUM_FEATURES, X_test, Y_test);

    // 2. Initialize Model and Optimizer
    // Debug set to 'true' to see your formatted epoch logs
    Model model(0.5, EPOCHS, true); 
    GradientDescent opt;

    // 3. Measure Training Time
    auto start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Training..." << std::endl;
    model.train(X_train, Y_train, opt);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 4. Evaluate on Unseen Data (Inference)
    double final_acc = model.test(X_test, Y_test);

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Training Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Final Test Accuracy: " << (final_acc * 100.0) << "%" << std::endl;

    // Assertions
    assert(final_acc > 0.85); // Model should easily learn this linear rule
    std::cout << "SUCCESS: Large scale test passed accuracy requirements." << std::endl;
}

int main() {
    try {
        run_large_test();
    } catch (const std::exception& e) {
        std::cerr << "Test suite failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}