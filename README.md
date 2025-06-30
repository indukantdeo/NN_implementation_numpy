# NN_implementation_numpy
This is the repository containing implementation of fully connected neural network using numpy. The code is part of the course assignment of UBC CPSC 532S.

## Create a virtual environment
```bash # 
load python 
pip install venv 
python -m venv venv 
source venv/bin/activate 
pip install -r requirements.txt 
```

## SRC
### FCN.py
Contains the implementation of fully connected layer
### loss.py
Mean squared error implementation
### activation.py
ReLu implementation

## Examples

### Example 1
```
 linear regression y = Wx + b
 N, Din, Dout = 4, 3, 2
 Training loop (plain SGD)
```

### Example 2
```
 3-layer fully-connected network: 10 → 32 → 16 → 1
 64 samples, 10-D input, 1-D output
 Training loop (plain SGD)
```


