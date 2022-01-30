import pandas as pd
import Tree as tree


class RandomForest:
    def __init__(
            self,
            trees_depth=0,
    ):
        self.trees = []
        self.trees_depth = trees_depth

    @staticmethod
    def most_frequent(List):
        return max(set(List), key=List.count)

    def predict(self, row: pd.Series, node: tree.Node):
        if node.node_type == 'leaf':
            return node.prediction

        else:
            if type(node.rule['value']) == str:
                if row[node.rule['column']] == node.rule['value']:
                    return self.predict(row, node.right)
                else:
                    return self.predict(row, node.left)
            else:
                if row[node.rule['column']] >= node.rule['value']:
                    return self.predict(row, node.right)
                else:
                    return self.predict(row, node.left)

    def predict_from_forest(self, row: pd.Series, classes):
        predictions = []
        for node in self.trees:
            pred = self.predict(row, node)
            predictions.append(pred)
        prediction = self.most_frequent(predictions)
        # print(f"prediction: {prediction}, value: {row[row.size - 1]}")
        if row[row.size - 1] == prediction:
            if row[row.size - 1] == classes[0]:
                return 'TN'
            else:
                return 'TP'
        else:
            if row[row.size - 1] == classes[0]:
                return 'FP'
            else:
                return 'FN'

    def predict_set(self, dataset: pd.DataFrame, classes):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        predictions_count = 0
        predicted_count = 0
        for index, row in dataset.iterrows():
            predict_result = self.predict_from_forest(row, classes)
            if predict_result == 'TN':
                true_negative = true_negative + 1
            if predict_result == 'TP':
                true_positive = true_positive + 1
            if predict_result == 'FN':
                false_negative = false_negative + 1
            if predict_result == 'FP':
                false_positive = false_positive + 1

        print(f"true positives: {true_positive}")
        print(f"true negative  {true_negative}")
        print(f"false positive: {false_positive}")
        print(f"false negative: {false_negative}")
