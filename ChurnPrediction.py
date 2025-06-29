import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# Step 1: Load data
try:
    data = pd.read_csv('Churn_Modelling.csv')  # Update path
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: File 'Churn_Modelling.csv' not found. Check path.")
    exit()

# Step 2: Prepare data
X = data[['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'IsActiveMember']]
y = data['Exited']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict and check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}%")

# Step 6: Plot feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
fig = px.bar(feature_importance, x='Feature', y='Importance', title='What Drives Customer Churn?')
fig.show()  # Opens in browser
fig.write_html('churn_plot.html')  # Saves as HTML