import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        best_feature = best_threshold = best_mse = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mse = np.mean((y[left_mask] - np.mean(y[left_mask])) ** 2)
                right_mse = np.mean((y[right_mask] - np.mean(y[right_mask])) ** 2)
                mse = (np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse) / len(y)

                if best_mse is None or mse < best_mse:
                    best_feature, best_threshold, best_mse = feature, threshold, mse

        if best_feature is None:
            return np.mean(y)

        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[X[:, best_feature] <= best_threshold], y[X[:, best_feature] <= best_threshold], depth + 1),
            'right': self._build_tree(X[X[:, best_feature] > best_threshold], y[X[:, best_feature] > best_threshold], depth + 1)
        }
        return tree

    def _predict_tree(self, tree, X):
        if isinstance(tree, (int, float)):
            return tree
        if X[tree['feature']] <= tree['threshold']:
            return self._predict_tree(tree['left'], X)
        return self._predict_tree(tree['right'], X)

    def fit(self, X, y):
        self.trees = []
        residuals = y.copy()

        for _ in range(self.n_estimators):
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)
            predictions = np.array([self._predict_tree(tree, x) for x in X])
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        return np.sum([[self._predict_tree(tree, x) for x in X] for tree in self.trees], axis=0) * self.learning_rate

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the model
gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
