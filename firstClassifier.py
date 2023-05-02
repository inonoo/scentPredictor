import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
from scent import*

class firstClassifer():
    def __init__(self):
        self.model = None
        self.columns = None

    def split(self, df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ''' Get training perfume and their basenotes
        :arg: df(pd.Dataframe): dataframe of scent and their basenote
        :return: [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_features, test features, train labels and test_labels
        '''
        features = df.drop('scents', axis=1)
        label = df['scents']
        # Extract features and labels
        features, label = df.drop('scents', axis=1), df[['scents']]
        # Split the data
        train_features, train_labels, test_features, test_labels = sklearn.model_selection.train_test_split(features, label, random_state=1)

        return (train_features.reset_index(drop=True),
                test_features.reset_index(drop=True),
                train_labels.reset_index(drop=True),
                test_labels.reset_index(drop=True))

    def train(self, features: pd.DataFrame, label: pd.Series) -> None:
        ''' Train a xgboost
            Args:
                features(pd.DataFrame): the features that will be used to predict the new price
                label(pd.series): column of the scent of the perfume
        '''
        self.model = xgb.XGBClassifier()
        self.model.fit(features, label)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        ''' Predict the new price of the scent brand
            Args:
                features(pd.Datafrome): data frame of the features matching the training columns

            Returns: the predicted scent of the brand

            '''
        return self.model.predict(features.astype(int))

    def assess(self, features: pd.DataFrame, labels: pd.Series)-> float:
        predicted_labels = self.predict(features)

        absolute_error = np.abs(labels - predicted_labels)
        mean_absolute_error = np.mean(absolute_error)
        print('Mean absolute error:', mean_absolute_error)
        return mean_absolute_error





