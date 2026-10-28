import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
df=pd.read_csv("Iris.csv")#relative path of the dataset file (.csv)
df=df.drop("Id", axis=1)
X=df.drop("Species", axis=1)
y=df["Species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)
accuracy=accuracy_score(pred,y_test)
print(accuracy)
joblib.dump(model,"model.pkl")
