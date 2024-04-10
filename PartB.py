import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

# connection to PostgreSQL database
engine = create_engine('postgresql://admin:1234@host:1435/data_mart')

# Query data from the FactTerrorismIncidents table
query = """
SELECT * FROM FactTerrorismIncidents
JOIN DateTimeDimension ON FactTerrorismIncidents.DateKey = DateTimeDimension.DateKey
JOIN LocationDimension ON FactTerrorismIncidents.LocationKey = LocationDimension.LocationKey
JOIN EconomicIndicatorsDimension ON FactTerrorismIncidents.EconomicIndicatorKey = EconomicIndicatorsDimension.EconomicIndicatorKey;
"""
df = pd.read_sql(query, engine)

# preprocessing
# Convert 'EconomicImpact' into two categories
df['EconomicImpact'] = df['EconomicImpact'].apply(lambda x: 'High' if x > 5 else 'Low')

# Defining X and y
X = df.drop('EconomicImpact', axis=1)
y = df['EconomicImpact']

# Splitting into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label='High'),
        'Recall': recall_score(y_test, y_pred, pos_label='High'),
        'Time to Construct': end_time - start_time
    })

#Results
results_df = pd.DataFrame(results)
print(results_df)

