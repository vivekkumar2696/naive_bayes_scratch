import pandas as pd

class NaiveBayes():
    def __init__(self):
        self.intermediate_dict = {}
        self.class_probablity = {}

    def train(self, X, y):
        pass

    def predict(self, X):
        output_prob = {}
        max_prob = 0.0
        output_label = None
        for class_key in self.class_probablity:
            output_prob[class_key] = self.class_probablity[class_key]
            for key in X:
                value = X[key]
                output_prob[class_key] *= self.intermediate_dict[key][value][class_key]
            if max_prob < output_prob[class_key]:
                output_label = class_key
                max_prob = output_prob[class_key]

        print(output_prob, output_label)
        return output_label

    def calc_class_probablity(self, df):
        intermediate_class_df = df.groupby(['Play-Tennis'])['Play-Tennis'].count()
        for class_col in intermediate_class_df.index.values:
            self.class_probablity[class_col] = intermediate_class_df[class_col]/intermediate_class_df.sum()

        print(self.class_probablity)

    def intermediate_class_data(self, df):
        # X['class'] = y
        columns = df.columns

        for col in columns:
            int_df = df.groupby(['Play-Tennis', col])[col].count()

            self.intermediate_dict[col] = {}
            for values in int_df.index.values.tolist():
                if(values[1] not in self.intermediate_dict[col]):
                    self.intermediate_dict[col][values[1]] = {}
                class_wise_sum = int_df[values[0]].sum()
                self.intermediate_dict[col][values[1]][values[0]] = int_df[values[0]][values[1]]/class_wise_sum
        print(self.intermediate_dict)
        self.calc_class_probablity(df)

if __name__ == "__main__":
    df = pd.read_csv("data/tennis_anyone.csv")
    naive_bayes = NaiveBayes()
    naive_bayes.intermediate_class_data(df)
    naive_bayes.predict({'Outlook':'Sunny', 'Temperature':'Cool', 'Humidity':'High', 'Wind':'Strong'})


