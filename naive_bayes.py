import pandas as pd
from sklearn.model_selection import train_test_split

class NaiveBayes():
    def __init__(self):
        self._intermediate_dict = {}
        self.class_probablity = {}

    def train(self, X, y):
        self._intermediate_class_data(X, y)

    def predict(self, X):
        """
        Perform classification on a dictionary on a test vector X

        Arguments:
            self : type
            X : array_like

        Returns:
            [array] -- [predicted labels]
        """
        output_prob = {}
        max_prob = 0.0
        output_labels = []
        for row in X:
            for class_key in self.class_probablity:
                output_prob[class_key] = self.class_probablity[class_key]
                for key in row:
                    value = row[key]
                    output_prob[class_key] *= (self._intermediate_dict[key][value][class_key] if class_key in self._intermediate_dict[key][value] else 0)
                if max_prob < output_prob[class_key]:
                    output_label = class_key
                    max_prob = output_prob[class_key]
            output_labels.append(output_label)
        return output_labels

    def _calc_class_probablity(self, df):
        intermediate_class_df = df.groupby(['Play-Tennis'])['Play-Tennis'].count()
        for class_col in intermediate_class_df.index.values:
            self.class_probablity[class_col] = intermediate_class_df[class_col]/intermediate_class_df.sum()

    def _intermediate_class_data(self, X, y):
        # X['class'] = y
        columns = X.columns

        df = X

        if type(y) == pd.core.frame.DataFrame:
            label_column = y.columns[0]
            df[label_column] = y[label_column]
        elif type(y) == pd.core.series.Series:
            label_column = y.name
            df[label_column] = y.values
        print(label_column)
        for col in columns:
            int_df = df.groupby([label_column, col])[col].count()

            self._intermediate_dict[col] = {}
            for values in int_df.index.values.tolist():
                if(values[1] not in self._intermediate_dict[col]):
                    self._intermediate_dict[col][values[1]] = {}
                class_wise_sum = int_df[values[0]].sum()
                self._intermediate_dict[col][values[1]][values[0]] = int_df[values[0]][values[1]]/class_wise_sum
        print(self._intermediate_dict)

        self._calc_class_probablity(df)

if __name__ == "__main__":
    df = pd.read_csv("data/tennis_anyone.csv")
    naive_bayes = NaiveBayes()
    y_label = 'Play-Tennis'
    y = df[[y_label]]
    X = df.drop(y_label, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    naive_bayes.train(X_train, y_train)
    print(naive_bayes.predict([{'Outlook':'Sunny', 'Temperature':'Cool', 'Humidity':'High', 'Wind':'Strong'}]))
