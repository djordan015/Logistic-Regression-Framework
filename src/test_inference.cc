#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include "model.h"

// Helper function to print results
void print_inference(const std::vector<double>& input, double prob, double threshold) {
    std::cout << "Input: [" << input[0] << ", " << input[1] << "] | "
              << "Prob: " << std::fixed << std::setprecision(4) << prob << " | "
              << "Class: " << (prob >= threshold ? "1" : "0") << std::endl;
}

void test_model_inference() {
    std::cout << "--- Starting Inference Test Suite ---" << std::endl;

    // 1. Setup & Train a model to learn a simple boundary (x + y > 0)
    std::vector<std::vector<double>> X_train = {
        {2.0, 2.0}, {1.0, 3.0},   // Clearly Class 1
        {-2.0, -2.0}, {-1.0, -3.0} // Clearly Class 0
    };
    std::vector<double> Y_train = {1.0, 1.0, 0.0, 0.0};
    
    double threshold = 0.5;
    Model model(X_train, Y_train, threshold, 0.1, 2000);
    model.train();

    // 2. Prepare Inference Data (Unseen by the model)
    std::vector<std::vector<double>> X_inference = {
        {1.5, 1.5},   // Should be Class 1
        {-1.5, -1.5}, // Should be Class 0
        {0.1, 0.1},   // Near boundary, but slightly Class 1
        {-0.1, -0.1}  // Near boundary, but slightly Class 0
    };

    LogitClassifier inference_engine;
    std::vector<double> results = inference_engine.forward_batch(
        X_inference, 
        model.get_weights(), 
        model.get_bias()
    );

    // 3. Assertions and Validation
    std::cout << "Testing unseen data points:" << std::endl;
    
    // Test Point 1: Strong Positive
    assert(results[0] > 0.5);
    print_inference(X_inference[0], results[0], threshold);

    // Test Point 2: Strong Negative
    assert(results[1] < 0.5);
    print_inference(X_inference[1], results[1], threshold);

    // Test Point 3: Probabilities are valid (0 <= p <= 1)
    for(double p : results) {
        assert(p >= 0.0 && p <= 1.0);
    }

    // 4. Test Edge Case: Consistency
    // The same input should result in the exact same probability
    double first_pass = inference_engine.forward_batch({{1.0, 1.0}}, model.get_weights(), model.get_bias())[0];
    double second_pass = inference_engine.forward_batch({{1.0, 1.0}}, model.get_weights(), model.get_bias())[0];
    assert(first_pass == second_pass);

    std::cout << "\nSUCCESS: Inference suite passed." << std::endl;
}

int main() {
    try {
        test_model_inference();
    } catch (const std::exception& e) {
        std::cerr << "Inference test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}