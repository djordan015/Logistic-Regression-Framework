<!--$https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/ -->
# 637 Logistic Regression Framework

## Project Structure

```
├── src/
│   ├── LogitClassifier.h     # Logistic Regression logic
│   ├── Model.h               # Abstract base classes for extensibility
│   ├── Optimizer.h           # Gradient Descent & optimization routines
│   ├── Types.h               # Custom type aliases 
│   ├── bindings.cc           # Pybind11 glue code
│   └── compile.sh            # One-touch build automation
├── python/                   # Python wrapper 
├── cli.py                    # Main entry point for model interaction
├── CMakeLists.txt            # Cross-platform build configuration
└── requirements.txt          # Python dependencies (e.g., NumPy, Pandas)
```

## Setup
Download repo 

Configure project using CMake

```
# From the project root
cmake -S . -B build
cmake --build build
```

Install required libraries
```
pip install -r requirements.txt
```

## Usage
``
python3 cli.py 
``