<!--$https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/ -->
# 637 Logistic Regression Framework
By: David Jordan and Rohit Patil

This Python framework allows for logistic regression model training and inference on ```.csv``` files. This framework works for data sets of varying dimensions.

This framework also enables weight saving and loading. 
- ```.json``` files are used for loading and saving weights

>Note: Data must be cleaned, normalized and scaled to achieve strong results<br>
>Note: Column order must remain the same between training and testing

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
1. Clone repo
```
git clone https://github.com/djordan015/Logistic-Regression-Framework.git
```

2. Add pybind11 submodule
```
git submodule add https://github.com/pybind/pybind11.git pybind11
```

3. Configure project using CMake

```
# From the project root
mkdir build
cd build

# from build/
cmake ..
make
cd..
```

4. Create virtual environment and Install required libraries
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Setup Complete**
## Usage
- Add your .csv file into project directory. (provide relative path in CLI)
- run the cli script
<br>
```
python3 cli.py 
```


## References
https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression
