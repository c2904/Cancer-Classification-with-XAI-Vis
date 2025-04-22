import pandas as pd
import math


class Preprocessor:
    @staticmethod
    def deleteRowIfColumnIsNan(data_frame: pd.DataFrame, column_name: str):
        data_frame = data_frame.dropna(subset=[column_name])
        return data_frame

