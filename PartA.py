import pandas as pd
import psycopg2
from sqlalchemy import create_engine

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

#preprocess step
df['NumberOfInjured'].fillna(df['NumberOfInjured'].median(), inplace=True)

# Normalizing the 'NumberOfDeaths' column
df['NumberOfDeaths'] = (df['NumberOfDeaths'] - df['NumberOfDeaths'].min()) / (df['NumberOfDeaths'].max() - df['NumberOfDeaths'].min())

# Onehot encoding the 'AttackTypeDimension'
df = pd.concat([df, pd.get_dummies(df['AttackTypeName'], prefix='AttackType')], axis=1)

# Dropping the original ' AttackTypeName' column
df.drop('AttackTypeName', axis=1, inplace=True)

####################################################
###Making the plots
####################################################

import matplotlib.pyplot as plt
import seaborn as sns

# scatter plot: GDPPerCapita vs NumberOfDeaths
sns.scatterplot(data=df, x='GDPPerCapita', y='NumberOfDeaths')
plt.title('GDP Per Capita vs. Number of Deaths due to Terrorism')
plt.xlabel('GDP Per Capita')
plt.ylabel('Number of Deaths')
plt.show()

# Boxplot for Economic Impact
sns.boxplot(data=df, y='EconomicImpact')
plt.title('Distribution of Economic Impact Scores')
plt.ylabel('Economic Impact Score')
plt.show()

# Histogram for NumberOfAttackers
plt.hist(df['NumberOfAttackers'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Number of Attackers')
plt.xlabel('Number of Attackers')
plt.ylabel('Frequency')
plt.show()

