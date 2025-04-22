import pandas as pd
import math


class Preprocessor:
    @staticmethod
    def deleteRowIfColumnIsNan(data_frame: pd.DataFrame, column_name: str):
        for index, row in data_frame.iterrows():
            if not (row[column_name] > 0):
                data_frame.drop(index, inplace=True)
        return data_frame

    @staticmethod
    def replaceNanValuesWithMedian(data_frame: pd.DataFrame):
        return data_frame.fillna(data_frame.median())

