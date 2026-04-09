#include <iostream>
#include <vector>
#include <cassert>
#include "model.h"
// this test suite was generated using gemini
void test_training_convergence() {
    std::cout << "Running Convergence Test..." << std::endl;

    // 1. Create a simple linearly separable dataset
    // Samples [1, 1] -> Class 1 | Samples [-1, -1] -> Class 0
    std::vector<std::vector<double>> X = {
        {1.0, 1.0}, {1.5, 2.0}, {2.0, 1.5},  // Class 1
        {-1.0, -1.0}, {-1.5, -2.0}, {-2.0, -1.5} // Class 0
    };
    std::vector<double> Y = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};

    // 2. Initialize Model: threshold=0.5, lr logic handled in schedule, 1000 epochs
    Model model(X, Y, 0.5, 0.01, 1000);

    // Capture initial state
    double initial_bias = model.get_bias();
    std::vector<double> initial_weights = model.get_weights();

    // 3. Train
    model.train();

    // 4. Verify weights have changed
    assert(model.get_bias() != initial_bias);
    assert(model.get_weights()[0] != initial_weights[0]);

    // 5. Manual Prediction Check
    // With this dataset, weights and bias should be positive
    LogitClassifier checker;
    std::vector<double> final_probs = checker.forward_batch(X, model.get_weights(), model.get_bias());
    
    bool improved = true;
    for(size_t i = 0; i < Y.size(); ++i) {
        // High values for class 1, low for class 0
        if(Y[i] == 1.0 && final_probs[i] <= 0.5) improved = false;
        if(Y[i] == 0.0 && final_probs[i] >= 0.5) improved = false;
    }

    if(improved) {
        std::cout << "SUCCESS: Model converged on a simple dataset." << std::endl;
    } else {
        std::cout << "FAILURE: Model did not classify simple points correctly." << std::endl;
    }
}

int main() {
    try {
        test_training_convergence();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}