import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Training data
# Any csv file as input 
data = {
    "YearsExperience": [1, 2, 3, 4, 5],
    "Salary": [30000, 40000, 50000, 60000, 70000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["YearsExperience"]]
y = df["Salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")