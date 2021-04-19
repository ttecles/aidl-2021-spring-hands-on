
# Session 2
Implement a full project training on [Chinese MNIST](https://www.kaggle.com/gpreda/chinese-mnist) with hyperparameter tuning.
## Installation
### With Conda
Create a conda environment by running
```
conda create --name aidl-session2 python=3.8
```
Then, activate the environment
```
conda activate aidl-session2
```
and install the dependencies
```
pip install -r requirements.txt
```
## Running the project

To run the project without hyperparameter tuning, run
```
python session-2/main.py
```

To run the project with hyperparameter tuning, run
```
python session-2/main_hyperparam_optimize.py
```

Windows pytorch with cuda support

```
pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```