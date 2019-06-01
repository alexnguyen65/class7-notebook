from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target

print(f'sklearn dataset X shape: {X.shape}')
print(f'sklearn dataset y shape: {y.shape}')
print(f'keys: {boston.keys()}')
print(f'data: {boston.data}')
print(f'target: {boston.target}')
print(f'feature_names: {boston.feature_names}')


