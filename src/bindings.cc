#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include <pybind11/numpy.h>
#include"model.h"
#include"LogitClassifier.h"
#include"Optimizer.h"
#include"Types.h"

namespace py = pybind11;

PYBIND11_MODULE(logistic_regression_cpu, m){
  py::class_<Model>(m, "Model")
    .def(py::init<double, double, int, bool>())
    .def("train", [](Model& self,
                 py::array_t<double, py::array::c_style | py::array::forcecast> X,
                 py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                 Optimizer& opt,
                 bool use_omp) {
        auto X_buf = X.request();
        auto Y_buf = Y.request();
                    
        if (X_buf.ndim != 2) throw std::runtime_error("X must be 2-D");
        if (Y_buf.ndim != 1) throw std::runtime_error("Y must be 1-D");
                    
        const int N = static_cast<int>(X_buf.shape[0]);
        const int M = static_cast<int>(X_buf.shape[1]);
        if (static_cast<int>(Y_buf.shape[0]) != N) {
            throw std::runtime_error("X and Y must have the same number of samples");
        }
      
        // MatrixView points directly into numpy's memory — no copy
        MatrixView X_view{ static_cast<const double*>(X_buf.ptr), N, M };
      
        // For Y, copy once into a std::vector (it's 1-D and small).
        // You could alternatively teach Model::train to accept a pointer + length for Y too.
        const double* Y_ptr = static_cast<const double*>(Y_buf.ptr);
        std::vector<double> Y_vec(Y_ptr, Y_ptr + N);
      
        self.train(X_view, Y_vec, opt, use_omp);
    })
    // Y_unseen is a non-const ref in C++; copy it so Python lists bind correctly
    .def("test", [](Model& self, const std::vector<std::vector<double>>& X, std::vector<double> Y){
      return self.test(X, Y);
    })
    .def("get_weights", &Model::get_weights, py::return_value_policy::copy)
    .def("get_bias", &Model::get_bias)
    .def("get_accuracy", &Model::get_accuracy)
    .def("set_epochs", &Model::set_epochs)
    .def("set_threshold", &Model::set_threshold)
    .def("set_learning_rate", &Model::set_learing_rate)
    .def("get_snapshot", &Model::get_snapshot)
    .def("load_snapshot", &Model::load_snapshot);

  py::class_<Model::ModelSnapshot>(m, "ModelSnapshot")
    .def_readwrite("weights", &Model::ModelSnapshot::weights)
    .def_readwrite("bias", &Model::ModelSnapshot::bias);

  py::class_<Optimizer>(m, "Optimizer");

  py::class_<SGD, Optimizer>(m, "SGD")
    .def(py::init<>());

  py::class_<GradientDescent, Optimizer>(m, "GradientDescent")
    .def(py::init<>());

  py::class_<LogitClassifier>(m, "LogitClassifier")
    .def(py::init<>())
    // weights is a non-const ref in C++; copy it so Python lists bind correctly
    .def("forward_batch", [](LogitClassifier& self, const std::vector<std::vector<double>>& X, std::vector<double> weights, double bias, bool flag){
      return self.forward_batch(X, weights, bias, flag);
    });

  py::class_<Gradients>(m, "Gradients")
    .def(py::init<>())
    .def_readwrite("dW", &Gradients::dW)
    .def_readwrite("dB", &Gradients::dB)
    .def_static("calculate_gradients", &Gradients::calculate_gradients);
}
