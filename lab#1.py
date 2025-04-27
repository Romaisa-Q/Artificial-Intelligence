import sklearn #machine learning
import numpy as np #handle numbers
import pandas as pd #read data/excel
from sklearn import linear_model #linear regression model
x = np.array([[5],[15],[25],[35],[45],[55]])
y = np.array([5,20,14,32,22,38])
modle = linear_model.LinearRegression()
modle.fit(x,y)
x_new = np.array([150]).reshape((-1,1)) #reshape((-1,1)) neccesssary bcz computer need 2d array.
y_new = modle.predict(x_new)
print(y_new) #output

import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_excel("data.xlsx")
x = df[["hours"]]
y = df[["score"]]

model = linear_model.LinearRegression()
model.fit(x, y)

x_new = np.array([[40]])
y_new = model.predict(x_new)

print(y_new)



