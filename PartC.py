import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# connection params
db_config = {
    "dbname": "your_database_name",
    "user": "your_username",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"  # default PostgreSQL port
}

#connection URL
connection_url =f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

# Creating  db engine to create queries on
engine = create_engine(connection_url)

# Query to select the data we need
query = """
SELECT * FROM FactTerrorismIncidents
JOIN LocationDimension ON FactTerrorismIncidents.LocationKey = LocationDimension.LocationKey
JOIN EconomicIndicatorsDimension ON FactTerrorismIncidents.EconomicIndicatorKey = EconomicIndicatorsDimension.EconomicIndicatorKey;
"""

# Importing data
df = pd.read_sql(query, engine)

# Feature selection for outlier detection assuming X is features
X = df[['NumberOfDeaths', 'TotalDamageCost', 'GDP', 'GDPPerCapita']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fitting OneClass SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(X_scaled)

# Predicting outliers
outliers = oc_svm.predict(X_scaled)
df['Outlier'] = outliers  # Adding  outliers to the df for analysis

# Display the number of outliers
print(f"Number of outliers detected: {(outliers == -1).sum()}")
