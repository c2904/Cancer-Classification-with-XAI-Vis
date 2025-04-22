import pandas as pd
import plotter as Plotter

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from DataManager import DataManager
from sklearn.model_selection import train_test_split

class Classifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.data_manager = DataManager()
        self.labels, self.data = self.data_manager.get_labels_and_data()


    def train_model(self):
        print("Data shape:", self.data.shape)
        print("Labels shape:", self.labels.shape)
        print("First rows of data:\n", self.data.head())
        print("First labels:\n", self.labels.head())

        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        self.model.fit(x_train, y_train)
        y_predict = self.model.predict(x_test)

        print("Accuracy Score:", accuracy_score(y_test, y_predict))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
        print("Classification Report:\n", classification_report(y_test, y_predict))

        return y_predict


    def predict(self, input_data):
        return self.model.predict(input_data)
