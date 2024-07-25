import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\ADISKA PUTRI AULIA\\Downloads\\onlinefoods.csv")
df
print(df.info())
print(df.describe())
print(df.isnull().sum())
sns.countplot(x='Output', data=df) 
plt.title('Distribution of Output')
plt.show()

sns.boxplot(x='Output', y='Age', data=df)
plt.title('Age Distribution by Output')
plt.show()

# Select only numeric columns for correlation matrix
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(df[numeric_columns].corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Handling missing values
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Encoding categorical variables
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']
numerical_features = ['Age', 'Family size', 'latitude', 'longitude']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Splitting the dataset into training and testing sets
X = df.drop('Output', axis=1)
y = df['Output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

# Load and preprocess data
df = pd.read_csv("C:\\Users\\ADISKA PUTRI AULIA\\Downloads\\onlinefoods.csv")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Encode categorical variables
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']
numerical_features = ['Age', 'Family size', 'latitude', 'longitude']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Split the dataset
X = df.drop('Output', axis=1)
y = df['Output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and preprocessor
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessor
model = joblib.load('random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Input form for user
st.title('Prediksi Output untuk Online Foods')

# Input features
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Prefer not to say'])
occupation = st.selectbox('Occupation', ['Employee', 'House wife', 'Self Employeed', 'Student'])
monthly_income = st.selectbox('Monthly Income', ['Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000', 'No Income'])
educational_qualifications = st.selectbox('Educational Qualifications', ['School', 'Graduate', 'Post Graduate', 'Ph.D', 'Uneducated'])
feedback = st.selectbox('Feedback', ['Positive', 'Negative'])
age = st.number_input('Age', min_value=0)
family_size = st.number_input('Family Size', min_value=0)
latitude = st.number_input('Latitude')
longitude = st.number_input('Longitude')

# Create DataFrame from inputs
user_input = pd.DataFrame({
    'Gender': [gender],
    'Marital Status': [marital_status],
    'Occupation': [occupation],
    'Monthly Income': [monthly_income],
    'Educational Qualifications': [educational_qualifications],
    'Feedback': [feedback],
    'Age': [age],
    'Family size': [family_size],
    'latitude': [latitude],
    'longitude': [longitude]
})

# Button to make prediction
if st.button('Predict'):
    try:
        # Apply preprocessing
        user_input_processed = preprocessor.transform(user_input)
        
        # Make prediction
        prediction = model.predict(user_input_processed)
        prediction_proba = model.predict_proba(user_input_processed)
        
        # Display prediction
        st.write('### Hasil Prediksi')
        st.write(f'Output Prediksi: {prediction[0]}')
        st.write(f'Probabilitas Prediksi: {prediction_proba[0]}')
    except ValueError as e:
        st.error(f"Error during preprocessing: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")