import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
import pickle

from country_transformer import CountryTransformer

import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('hotel_booking.csv')

# Find cumulative bookings by same customer
#cum_bookings = df.groupby('name')['hotel'].count().reset_index()
#cum_bookings.columns = ['name', 'cum_bookings']

# Find Countries to keep
country_counts = df.groupby('country')['hotel'].count().reset_index()
country_counts.columns = ['country', 'country_counts']
country_counts['country_grouped'] = country_counts['country']
country_counts.loc[country_counts['country_counts'] < 1000, 'country_grouped'] = 'Others'

# Columns to drop
columns_to_drop = ['arrival_date_year', 'reservation_status_date', 'reservation_status','assigned_room_type','name', 'email','phone-number', 'credit_card','agent', 'company']

df.drop(columns=columns_to_drop, inplace=True)

# Pipeline for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', MinMaxScaler())  
])

# Pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, selector(dtype_include=np.number)),
        ('cat', categorical_transformer, selector(dtype_include=object))
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('country_transform', CountryTransformer(country_counts=country_counts)),
    ('preprocessor', preprocessor),  
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the model
X_train, X_test, y_train, y_test = train_test_split(df.drop('is_canceled', axis=1), df['is_canceled'], test_size=0.3, random_state=42)

model.fit(X_train, y_train)

with open('baseline_model.pkl', 'wb') as f:
    pickle.dump({'country_counts': country_counts, 'model': model}, f)
