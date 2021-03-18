import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7.0, 6.0)
d = pd.read_csv("http://bit.ly/w-data")

d1=d.copy()
d1=pd.DataFrame(d1)
Y= d1[['Scores']]

X= d1[['Hours']]

plt.scatter(X, Y)
plt.grid(color="Y")
plt.title("Hours vs Percentage")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, Y)
plt.grid()
plt.title("Hours vs Percentage")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.plot(X, line, color='blue') # predicted
plt.show()

y_pred = regressor.predict(X_test)
y_pred

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("Score is:",score)

h=[9.25]
p=regressor.predict([h])
print("Given Hours:",h[0])
print("Predicted Score of the Student:",p[0][0])
