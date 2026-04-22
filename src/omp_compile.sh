# Compile the original convergence test
# g++ -O3 -I. test.cc -o test_convergence

# Compile the large-scale inference test
# g++ -O3 -I. test_inference_large.cc -o test_inference

g++ test_new.cc -o test_omp -fopenmp -O3 