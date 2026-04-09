#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <random>
#include "model.h"

/**
 * Generates synthetic data for binary classification.
 * Rule: If sum(features) > 0, Label = 1.0, else 0.0.
 */
void generate_synthetic_data(int samples, int features, 
                             std::vector<std::vector<double>>& X, 
                             std::vector<double>& Y) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
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

void test_large_scale_inference() {
    const int NUM_FEATURES = 10;
    const int NUM_SAMPLES = 20000;
    const int TEST_SAMPLES = 100; // Unseen samples for inference

    std::cout << "--- Large Scale Inference Test ---" << std::endl;
    std::cout << "Generating " << NUM_SAMPLES << " samples with " << NUM_FEATURES << " features..." << std::endl;

    // 1. Generate Training Data
    std::vector<std::vector<double>> X_train;
    std::vector<double> Y_train;
    generate_synthetic_data(NUM_SAMPLES, NUM_FEATURES, X_train, Y_train);

    // 2. Initialize and Train
    // Note: Threshold 0.5, LR logic in schedule, 500 epochs for speed
    Model model(X_train, Y_train, 0.5, 0.01, 500);
    
    std::cout << "Training model (this may take a moment)..." << std::endl;
    model.train(); //

    // 3. Generate Unseen Test Data for Inference
    std::vector<std::vector<double>> X_test;
    std::vector<double> Y_test;
    generate_synthetic_data(TEST_SAMPLES, NUM_FEATURES, X_test, Y_test);

    // 4. Run Inference
    LogitClassifier inference_engine;
    std::vector<double> probs = inference_engine.forward_batch(
        X_test, 
        model.get_weights(), 
        model.get_bias()
    ); //

    // 5. Evaluate Performance
    int correct = 0;
    for (int i = 0; i < TEST_SAMPLES; ++i) {
        double prediction = (probs[i] >= 0.5) ? 1.0 : 0.0;
        if (prediction == Y_test[i]) {
            correct++;
        }
    }

    double accuracy = (static_cast<double>(correct) / TEST_SAMPLES) * 100.0;
    std::cout << "Inference Accuracy on Unseen Data: " << accuracy << "%" << std::endl;

    // Reliability Assertions
    assert(probs.size() == TEST_SAMPLES);
    assert(accuracy > 80.0); // Simple linear rules should be learned easily
    
    std::cout << "SUCCESS: Large scale inference test passed." << std::endl;
}

int main() {
    try {
        test_large_scale_inference();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}