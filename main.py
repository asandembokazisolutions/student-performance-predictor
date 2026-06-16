import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("StudentsPerformance.csv")

# Features (input)
X = data[['gender', 'race/ethnicity', 'parental level of education',
          'lunch', 'test preparation course', 'reading score', 'writing score']]

# Target (output)
y = data['math score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical columns (LinearRegression requires numeric input)
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education',
                    'lunch', 'test preparation course']
numerical_cols   = ['reading score', 'writing score']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

# Build pipeline: preprocessor + model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R² :", round(r2_score(y_test, predictions), 4))
print("Predictions:", predictions[:5].round(1))
print("Actual:     ", y_test[:5].values)
