#include <vector>
#include <cmath>
#include <stdexcept>



class Optimizer{
    private:
        std::vector<double> wieghts;
        std::vector<double> bias;

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

        std::vector<double> calculate_gradients(){
            std::vector<double> gradients;

            return gradients;
        }

        double get_accurcay(){
            double accuracy = 0.0;

            return accuracy;
        }
};