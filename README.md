# Car Price Prediction Model

This project involves developing a neural network model to predict car prices based on various features such as years, kilometers driven, rating, condition, economy, top speed, horsepower, and torque. The model is built and trained using TensorFlow and Keras.

## Libraries Used

- TensorFlow
- Keras
- Pandas
- Seaborn
- Matplotlib
- NumPy

## Data Preparation

1. **Load Data**: The dataset is loaded using Pandas.
2. **Visualization**: Seaborn's pairplot is used to visualize relationships between features.
3. **TensorFlow Conversion**: Data is converted to TensorFlow tensors and shuffled.
4. **Splitting Data**: Data is split into training, validation, and test sets.
5. **Normalization**: Features are normalized using TensorFlow's `Normalization` layer.

## Model Architecture

The neural network is designed with the following layers:
- InputLayer
- Normalization layer
- Dense layers with ReLU activation
- Output layer with a single neuron for regression

## Model Compilation and Training

- **Optimizer**: Adam with a learning rate of 0.01
- **Loss Function**: MeanAbsoluteError
- **Metrics**: Accuracy

The model is trained for 1000 epochs with early stopping based on validation loss to prevent overfitting.

## Evaluation and Visualization

- **Evaluation**: The model's performance is evaluated on the test set.
- **Visualization**: Training and validation loss, as well as RMSE, are plotted. Predicted car prices are compared with actual prices using bar charts.

## Usage

To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/hyperion912/CarPricePrediction-using-DL.git
2. Install the required libraries:
```
  pip install tensorflow pandas seaborn matplotlib numpy
```
3. Run the script:
bash
```
  python car_price_prediction.py
```
##Results
The model effectively predicts car prices based on the given features. The performance is visualized through loss and RMSE plots, and predicted prices are compared against actual prices using bar charts.

