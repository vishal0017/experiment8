import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model=joblib.load("model.pkl")

test_df=pd.read_csv("Iris.csv")
test_df=test_df.drop("Id", axis=1)
X=test_df.drop("Species",axis=1)
y=test_df["Species"]

pred=model.predict(X)
accuracy=accuracy_score(pred,y)
print(accuracy)

#Test cases
assert accuracy>=0.98
assert len(pred)==len(y)
print("All test cases passed and model is ready to deploy")