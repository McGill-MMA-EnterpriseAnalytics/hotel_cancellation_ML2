from sklearn.base import BaseEstimator, TransformerMixin

class CountryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, country_counts):
        self.country_counts = country_counts

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.merge(self.country_counts[['country', 'country_grouped']], on='country', how='left')
        X = X.drop('country', axis=1)
        X['country_grouped'] = X['country_grouped'].fillna('Others')
        X.rename(columns={'country_grouped': 'country'}, inplace=True)
        return X
