

from pathlib import Path
import pandas as pd
import joblib

data_dir = '/gcs/marathon-time/processed' # Replace with data dir
data_dir = Path(data_dir)

data = pd.read_csv(data_dir / 'train.csv')



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
# Prepare data
X = data[['wall_21', 'km_per_week']]
y = data['marathon_time']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

joblib.dump(model, '/gcs/marathon-time/model-output/model.joblib') # Save the jobli model to GCS

