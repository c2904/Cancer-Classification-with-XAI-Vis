import pandas as pd
import numpy as np
from sklearn import preprocessing
from Preprocessor import Preprocessor

class DataManager:
    label_col = 'Overall Survival'

    data_frame = pd.DataFrame()
    data = pd.DataFrame()
    labels = pd.DataFrame()

    def __init__(self):
        self.__load_file()
        self.__process_data()
        self.__split_labels_from_data()

    def __load_file(self):
        dim_list = ['Overall Survival']
        self.data_frame = pd.read_csv("/home/celine/Desktop/Cancer-Classification-with-XAI-Vis/hcai_data-main/difg_glass_clinical_data.tsv", usecols=dim_list, sep='\t')
        print(self.data_frame.head())


    def __set_survival_groups_as_label(self):
        for i, row in self.data_frame.iterrows():
            if self.data_frame.loc[i][self.label_col] == 1:
                self.data_frame.at[i, self.label_col] = 1 #Deceased
            elif self.data_frame.loc[i][self.label_col] == 0:
                self.data_frame.at[i, self.label_col] = 2 #Living

    def __process_data(self):
        # replace nan values with mean of column
        self.data_frame = Preprocessor.replaceNanValuesWithMedian(data_frame=self.data_frame)

        self.__set_survival_groups_as_label()
        self.__categorical_data_to_numerical()

    def __categorical_data_to_numerical(self):
        dataframe_copy = self.data_frame.select_dtypes(include=['object']).copy()

        # join cols with int and float values
        int_dataframe = self.data_frame.select_dtypes(include=['int64']).copy()
        float_dataframe = self.data_frame.select_dtypes(include=['float64']).copy()
        dataframe_int_float = pd.concat([float_dataframe, int_dataframe], axis=1)

        le = preprocessing.LabelEncoder()
        dataframe_categorical = dataframe_copy.astype(str).apply(le.fit_transform)

        self.data_frame = pd.concat([dataframe_int_float, dataframe_categorical], axis=1)

    def get_whole_dataframe(self):
        return self.data_frame

    def __split_labels_from_data(self):
        dataframe = self.data_frame.copy()
        self.labels = dataframe.pop(self.label_col)
        self.data = dataframe

    def get_labels_and_data(self):
        return self.labels, self.data
