import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = load_boston()
X = boston.data
y = boston.target
columns_names = boston.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
lm = LinearRegression()
lm.fit(X_train, y_train)

print(f'Intercept: {lm.intercept_}')
print(f'Coefficients: {lm.coef_}')
#print(f'Named Coefficients: {pd.DataFrame(lm.coef_, columns_names)}')
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, columns_names)}")

