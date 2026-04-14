import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the trained model
model = joblib.load('models/model.pkl')

# Load the test dataset
df_test = pd.read_csv('data/diabetes.csv')

# Split the test data into features and target variable
X = df_test.drop('Outcome', axis=1)
y = df_test['Outcome']

# Make predictions on the test set
y_pred = model.predict(X)

# Evaluate the model's performance
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")

if accuracy < 0.65:
    raise Exception("Model accuracy is below the acceptable threshold of 0.65. Please retrain the model.")