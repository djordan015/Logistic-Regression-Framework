#ifndef MODEL_H
#define MODEL_H


#include <vector>
#include <cmath>
#include <stdexcept>
#include <Optimizer.h>
#include <LogitClassifier.h>
#include <Types.h>


class Model{
    private:
        std::vector<std::vector<double>> X_train;
        std::vector<double> Y_train;
        int epochs;
    
        LogitClassifier classifier;
        Optimizer optimizer;

        struct Gradients{ 
            std::vector<double> dW;
            double dB;
        }

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

        double get_accuracy(){
            double accuracy = 0.0;

            return accuracy;
        }

        double learning_rate_schedule(int epoch){
            if(epoch < 10000)
                return 0.1;
            else if(epoch < 20000)
                return 0.01;
            return 0.001;
        }

    public:
        Model(std::vector<std::vector<double>> X, 
                std::vector<double> Y, 
                double lr, 
                int ep) 
                : X_train(X), 
                Y_train(Y), 
                epochs(ep), 
                classifier(X_train[0].size()), // Init classifier with number of features
                optimizer(lr) // Init optimizer with learning rate
            {

                if (X.size() != Y.size()) {
                    throw std::invalid_argument("Features and targets must have same number of samples.");
                }
            }

        void train(){
            for (int epoch = 0; epoch < epochs; ++epoch) {
                // 1. Get Predictions (Forward Pass)
                std::vector<double> probs = classifier.forward_batch(X_train);

                // 2. Calculate Loss
                double current_loss = binary_cross_entropy(probs, Y_train);

                // 3. Calculate Gradients
                std::vector<double> dW = compute_gradients(probs); 

                // 4. Update Weights (The Optimizer's only job)
                double lr = learning_rate_schedule(epoch);
                optimizer.apply_step(classifier.get_weights(), dW, lr);
                
                // 5. Check Accuracy (Optional logging)
                if (epoch % 100 == 0) {
                    double acc = get_accuracy(probs, Y_train);
                    // Log: Epoch, Loss, Accuracy
                }
            }
        }
    
};

#endif