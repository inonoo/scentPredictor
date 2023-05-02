import pandas as pd
from sklearn.preprocessing import LabelBinarizer

class Scent():
    def __init__(self):
        self.features = None
        self.labels = None

    def process(data: pd.DataFrame) -> pd.DataFrame:
        '''
        This function will clean up the data
        param
            pd: The perfume dataframe
        :return:
            pd: Perfume dataframe with scents and the base notes.

        '''
        data.drop([480])  # this is the data point that are in the kids unisex department
        # this ensures we only have scents have brandnames and are for women, men, and unisex
        #select only the scents with at most 5 basenotes
        # only want scents 'floral', 'fruity'
        data_bn = data[['V1', 'V2', 'V3', 'V4', 'V5', 'scents']]
        # change words to lower case
        data_bn = data_bn.apply(lambda x: x.astype(str).str.lower())
        data_bn = data_bn.loc[(data_bn['scents'] == 'floral') | (data_bn['scents'] == 'fruity')]
        # use onehot encoding to find the presence and abscence of unique basenotes
        label_binarizer = LabelBinarizer()
        label_binarizer_output = label_binarizer.fit_transform(data_bn['V1'])
        # creating a data frame from the object
        result_df1 = pd.DataFrame(label_binarizer_output,
                                  columns=label_binarizer.classes_)
        label_binarizer = LabelBinarizer()
        label_binarizer_output = label_binarizer.fit_transform(data_bn['V2'])
        # creating a data frame from the object
        result_df2 = pd.DataFrame(label_binarizer_output,
                                  columns=label_binarizer.classes_)
        label_binarizer = LabelBinarizer()
        label_binarizer_output = label_binarizer.fit_transform(data_bn['V3'])
        # creating a data frame from the object
        result_df3 = pd.DataFrame(label_binarizer_output,
                                  columns=label_binarizer.classes_)
        label_binarizer = LabelBinarizer()
        label_binarizer_output = label_binarizer.fit_transform(data_bn['V4'])
        # creating a data frame from the object
        result_df4 = pd.DataFrame(label_binarizer_output,
                                  columns=label_binarizer.classes_)
        label_binarizer = LabelBinarizer()
        label_binarizer_output = label_binarizer.fit_transform(data_bn['V5'])
        # creating a data frame from the object
        result_df5 = pd.DataFrame(label_binarizer_output,
                                  columns=label_binarizer.classes_)
        #create a new dataframe
        new_df = pd.merge(result_df1, result_df2)
        new_df = new_df.merge(result_df3)
        new_df = new_df.merge(result_df4)
        new_df = new_df.merge(result_df5)
        data_bn = pd.merge(new_df, data_bn)

        #return new dataframe
        return data_bn

    def read_data(path: str) -> pd.DataFrame:
        '''
        Args:
            path(str): the path name

        Returns: the perfume dataset and cleans it is up

        '''
        df = pd.read_csv(path, index_col=0)
        features = df.drop('scents', axis=1)
        label = df['scents']
        return df.apply(process)


if __name__ == "__main__":
    df = read_data("/Users/rura/Desktop/noon_perfumes_dataset.csv")
    print(df)
