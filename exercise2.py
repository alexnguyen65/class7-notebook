from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
X = boston.data
y = boston.target

print(f'sklearn dataset X shape: {X.shape}')
print(f'sklearn dataset y shape: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")



