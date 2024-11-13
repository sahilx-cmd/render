import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('D:/AI3103/ca2project/model_training/labeling - Sheet.csv')

# Convert non-numeric values to NaN and handle missing values
data['eyetoeye'] = pd.to_numeric(data['eyetoeye'], errors='coerce')
data = data.dropna()

# Prepare features and target variable
X = data[['eyetoeye', 'eyetomouth']]
y = data['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
gb = GradientBoostingClassifier()
knn = KNeighborsClassifier()
svc = SVC(probability=True, random_state=42)

# Train individual models
gb.fit(X_train, y_train)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

# Predict and calculate accuracies for individual models
gb_pred = gb.predict(X_test)
knn_pred = knn.predict(X_test)
svc_pred = svc.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
svc_accuracy = accuracy_score(y_test, svc_pred)

print(f'Gradient Boosting Accuracy: {gb_accuracy}')
print(f'K-Nearest Neighbors Accuracy: {knn_accuracy}')
print(f'Support Vector Classifier Accuracy: {svc_accuracy}')

# Create an ensemble model with soft voting
ensemble_model = VotingClassifier(estimators=[
    ('gb', gb),
    ('knn', knn),
    ('svc', svc)
], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Calibrate the ensemble model using isotonic regression for more accurate probabilities
calibrated_model = CalibratedClassifierCV(ensemble_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# Evaluate calibrated model accuracy
y_pred = calibrated_model.predict(X_test)
calibrated_accuracy = accuracy_score(y_test, y_pred)

print(f'Calibrated Ensemble Model Accuracy: {calibrated_accuracy}')

# Save the calibrated ensemble model as a pickle file
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(calibrated_model, f)
