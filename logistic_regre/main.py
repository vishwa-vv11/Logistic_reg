#import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score
path=r"C:\Users\darkp\OneDrive\Desktop\vsc_projects\machine_learning\logistic regression\logistic_reg.csv"
df = pd.read_csv(path)
y=df["target"]
x=df.drop("target",axis=1)
print(x,y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

