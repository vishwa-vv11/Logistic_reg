import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
filepath=r"C:\Users\darkp\OneDrive\Desktop\vsc_projects\machine_learning\Linear_regression\linear_reg.csv"
df = pd.read_csv(filepath)
encoder=LabelEncoder()
df["Gender"]=encoder.fit_transform(df["Gender"])
y=df["Salary"]
x=df.drop("Salary",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print("MSE:",mse)
import numpy as np
from sklearn.metrics import mean_squared_error
# Baseline: predict mean salary for all test samples
baseline_pred = np.full_like(y_test, y_test.mean())
# Compute baseline MSE
baseline_mse = mean_squared_error(y_test, baseline_pred)
print("Baseline MSE:", baseline_mse)
input=np.array([[23,0]])
model.predict(input)
