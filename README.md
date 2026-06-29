# Time-Series Forecasting with Convolutional Position-Aware Attention Transformer

This project implements a time-series forecasting pipeline using TensorFlow/Keras. The code loads multivariate time-series data from a CSV file, converts it into supervised learning samples using a look-back window and prediction horizon, normalizes the data, and trains forecasting models.

The main model is a custom Transformer-style neural network that uses multi-head convolutional position-aware attention. A simple linear baseline model is also included for comparison.

## Overview

The pipeline performs the following steps:

1. Load a multivariate time-series dataset from a CSV file.
2. Convert the raw sequence into input-output forecasting samples.
3. Split the samples into training, validation, and test sets.
4. Normalize the input data using statistics from the training set.
5. Train a custom Transformer-based forecasting model.
6. Evaluate the model using MAE and RMSE.
7. Train and evaluate a simple dense linear baseline model.

## Dataset

The code expects a CSV file containing multivariate time-series data.

Each row represents one time step, and each column represents one variable or sensor.

Example dataset path used in the code:

```python
dp = '/content/drive/MyDrive/TransformerTimeSeriesDatasets/traffic.txt'
```

Other possible datasets mentioned in the code include:

```python
exchange_rate.txt
electricity.txt
traffic.txt
```

The code is currently written for use in Google Colab and mounts Google Drive before loading the dataset.

## Forecasting Setup

The forecasting task uses:

```python
lookback_window = 96
prediction_interval = 720
```

This means the model receives the previous 96 time steps as input and predicts selected future time steps within the next 720 time steps.

The target output is extracted at three forecast horizons:

```python
[96, 336, 720]
```

In zero-based indexing, these correspond to:

```python
[95, 335, 719]
```

So, for each input sequence, the model predicts values at three future time horizons.

## Data Preparation

The `DataLoader` class handles dataset preparation.

Given a time-series matrix of shape:

```text
time_steps × num_features
```

it creates:

```text
X: num_samples × lookback_window × num_features
Y: num_samples × 3 × num_features
```

The code then splits the dataset into:

```text
70% training
15% validation
15% testing
```

The input features are normalized using the training-set mean and standard deviation. The same normalization parameters are then applied to validation and test data.

The training data is also shuffled before training.

## Model 1: Convolutional Position-Aware Attention Transformer

The main model uses a custom attention layer called:

```python
multiHeadConvPaAttentionM2
```

This layer applies convolution before computing query, key, and value representations for each attention head.

The Transformer block is implemented as:

```python
TransformerBlockConvPaAttM2
```

Each block contains:

* Multi-head convolutional position-aware attention
* Dropout
* Layer normalization
* Feed-forward dense network

The model uses three Transformer blocks:

```python
num_transformer_blocks = 3
```

Main hyperparameters:

```python
num_heads = 6
ff_dim = 64
tconv_dim = 1
tconv_stride = 1
```

The model architecture is:

```text
Input
 → Dense embedding layer
 → 3 custom Transformer blocks
 → Global average pooling
 → Dense layer
 → Dropout
 → Output dense layer
```

The model is compiled with:

```python
optimizer = 'adam'
loss = 'mse'
metrics = ['mse']
```

## Learning Rate Schedule

A learning rate scheduler is used during training.

For the Transformer model, the initial learning rate is:

```python
0.001
```

and it decays by a factor of `0.75` every 50 epochs.

## Training the Transformer Model

The Transformer model is trained using:

```python
batch_size = 128
epochs = 10
```

Validation is performed using the test set in the current code:

```python
validation_data=(X_test, Y_test)
```

If you want a cleaner experimental setup, this can be changed to:

```python
validation_data=(X_val, Y_val)
```

## Evaluation

After training, the model predicts on the test set.

The following evaluation metrics are computed:

```python
MAE  = mean absolute error
RMSE = root mean squared error
```

The code also measures inference time using:

```python
st = time.time()
Y_pred = stdtr_model.predict(X_test, verbose=0)
et = time.time()
```

## Model 2: Linear Baseline

The code also includes a simple dense baseline model.

This model flattens the input sequence and directly maps it to the prediction output using one dense layer.

The baseline architecture is:

```text
Input
 → Dense output layer
```

It is compiled with:

```python
optimizer = 'adam'
loss = 'mse'
metrics = ['mse', 'mae']
```

The baseline model is trained for:

```python
epochs = 350
batch_size = 128
```

Its learning rate starts at:

```python
0.01
```

and decays by a factor of `0.75` every 50 epochs.

## Requirements

The main Python packages used are:

```text
numpy
pandas
matplotlib
seaborn
tensorflow
scikit-learn
scipy
```

The code is designed to run in Google Colab, especially because it uses:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## How to Run

1. Open the notebook or script in Google Colab.

2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Place the dataset file in your Google Drive.

4. Update the dataset path:

```python
dp = '/content/drive/MyDrive/TransformerTimeSeriesDatasets/traffic.txt'
```

5. Set the forecasting configuration:

```python
lookback_window = 96
prediction_interval = 720
```

6. Run the data preparation cells.

7. Train the Transformer model.

8. Evaluate MAE, RMSE, and inference time.

9. Optionally train the linear baseline model for comparison.

## Output

The code prints:

* Model summary
* Training progress
* Inference time
* Test MAE
* Test RMSE

Example output values are printed using:

```python
print(et - st)
print(test_mae)
print(test_se)
```

where `test_se` represents RMSE.

## Notes

* The current code uses the test set as validation data during training. For proper model selection, it is better to use the validation set and reserve the test set only for final evaluation.
* The output target contains predictions at three selected future horizons rather than the full prediction interval.
* The Transformer model expects input shape:

```text
samples × lookback_window × num_features
```

* The linear baseline expects flattened input shape:

```text
samples × flattened_features
```
