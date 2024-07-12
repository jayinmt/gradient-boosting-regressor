# Gradient Boosting Regressor

This project implements a simple Gradient Boosting Regressor from scratch using Python.

## Features

- Custom implementation of Gradient Boosting for regression tasks
- Simple decision tree as base learner
- Configurable number of estimators, learning rate, and maximum tree depth
- Example usage with synthetic data

## Requirements

- Python 3.7+
- NumPy
- scikit-learn (for data generation and evaluation purposes only)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/jaydxyz/gradient-boosting-regressor.git
   cd gradient-boosting-regressor
   ```

2. Install the required packages:
   ```
   pip install numpy scikit-learn
   ```

## Usage

The main script `gradient_boosting.py` contains the `GradientBoostingRegressor` class and a sample usage. To run the script:

```
python gradient_boosting.py
```

This will train the model on synthetic data and print the Mean Squared Error of the predictions.

To use the `GradientBoostingRegressor` in your own projects:

```python
from gradient_boosting import GradientBoostingRegressor

# Create and train the model
gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

# Make predictions
y_pred = gb.predict(X_test)
```

## How It Works

1. The algorithm starts with an initial prediction (the mean of the target values).
2. For each iteration:
   - It calculates the residuals (difference between the true values and the current predictions).
   - It fits a decision tree to these residuals.
   - It updates the predictions by adding the scaled (by learning rate) predictions of this tree.
3. The final prediction is the sum of the initial prediction and all the scaled tree predictions.

## Limitations

This implementation is simplified for educational purposes and may not perform as well as optimized libraries like XGBoost or LightGBM. It lacks advanced features such as:

- Column subsampling
- Regularization
- Handling of categorical variables
- Advanced tree-building algorithms

## Future Improvements

- Implement feature importance calculation
- Add support for classification tasks
- Optimize performance with Cython or Numba
- Implement early stopping based on validation error

## Contributing

Contributions to improve the implementation or extend its functionality are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes.

## License

This project is open source and available under the [MIT License](LICENSE).
