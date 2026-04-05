#ifndef TYPES_H
#define TYPES_H

#include <vector>

struct Gradients {
    std::vector<double> dW;
    double dB;
};

#endif