import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path = r"C:\Users\darkp\OneDrive\Desktop\vsc_projects\machine_learning\task3\Housing.csv"
df = pd.read_csv(path)

encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[["furnishingstatus"]])
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(["furnishingstatus"]))
df = pd.concat([df, encoded_df], axis=1)
df = df.drop("furnishingstatus", axis=1)

label_encoder = LabelEncoder()
cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in cols:
    df[col] = label_encoder.fit_transform(df[col])

y = df["price"]
x = df.drop("price", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)

coefficients = pd.DataFrame({'Feature': x.columns, 'Coefficient': model.coef_})
print("Intercept:", model.intercept_)
print(coefficients)

plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()
