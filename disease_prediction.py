import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = {
    'Patient': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
    'Age': [55, 60, 45, 50, 48, 52],
    'Chol': [250, 240, 230, 220, 210, 215],
    'BP': [140, 150, 130, 120, 125, 135],
    'Target': [1, 1, 1, 0, 0, 0] # 1 = Disease, 0 = No Disease
}
df = pd.DataFrame(data)

X = df[['Age', 'Chol', 'BP']]
y = df['Target']


model = GaussianNB()
model.fit(X, y)


new_patient = pd.DataFrame([[54, 245, 138]], columns=['Age', 'Chol', 'BP'])
prediction = model.predict(new_patient)
probabilities = model.predict_proba(new_patient)

print(f"--- Disease Prediction System ---")
print(f"Patient Data: Age=54, Chol=245, BP=138")
print(f"Prediction: {'Heart Disease Present (1)' if prediction[0] == 1 else 'No Disease (0)'}")
print(f"Probabilities -> No Disease: {probabilities[0][0]:.5f}, Disease: {probabilities[0][1]:.5f}")
