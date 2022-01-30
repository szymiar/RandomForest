import random
import pandas as pd
import numpy as np
from collections import Counter
import timeit


class Node:
    def __init__(
            self,
            Y: list,
            X: pd.DataFrame,
            max_depth=None,
            depth=None,
            node_type=None,
            rule=None,
    ):

        self.Y = Y
        self.X = X

        self.max_depth = max_depth if max_depth else 4

        self.depth = depth if depth else 0

        self.features = list(self.X.columns)

        self.node_type = node_type if node_type else 'root'

        self.rule = rule if rule else None

        self.counts = Counter(Y)

        self.gini_impurity = self.get_GINI()

        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        prediction = None
        if len(counts_sorted) > 0:
            prediction = counts_sorted[-1][0]

        self.prediction = prediction

        self.n = len(Y)

        self.left = None
        self.right = None

    def get_GINI(self):
        keys = list(self.counts.keys())
        if len(keys) == 1:
            return 0
        y1_count, y2_count = self.counts[keys[0]], self.counts[keys[1]]

        return self.GINI_impurity(y1_count, y2_count)

    @staticmethod
    def GINI_impurity(y1_count: int, y2_count: int) -> float:
        if y1_count is None:
            y1_count = 0
        if y2_count is None:
            y2_count = 0
        n = y1_count + y2_count
        if n == 0:
            return 0.0
        p1 = y1_count / n
        p2 = y2_count / n
        gini = 1 - (p1 ** 2 + p2 ** 2)
        return gini

    def grow_tree(self):
        df = self.X.copy()
        df['Y'] = self.Y
        if self.depth < self.max_depth and self.gini_impurity != 0 and self.node_type != 'leaf':

            if self.rule is None:
                new_rules = Node.choose_best_column(df, self.get_GINI())
                if new_rules is not None:
                    self.rule = {
                        'column': new_rules['column'],
                        'value': new_rules['divider']
                    }
                else:
                    self.node_type = 'leaf'
                    return

            if type(self.rule['value']) == str:
                left_df, right_df = df[df[self.rule['column']] != self.rule['value']].copy(), df[
                    df[self.rule['column']] == self.rule['value']].copy()
            else:
                left_df, right_df = df[df[self.rule['column']] < self.rule['value']].copy(), df[
                    df[self.rule['column']] >= self.rule['value']].copy()

            left = Node(
                left_df['Y'].values.tolist(),
                left_df[self.features],
                depth=self.depth + 1,
                max_depth=self.max_depth,
                node_type='left_node'
            )
            if left.gini_impurity == 0:
                left.node_type = 'leaf'

            self.left = left
            self.left.grow_tree()

            right = Node(
                right_df['Y'].values.tolist(),
                right_df[self.features],
                depth=self.depth + 1,
                max_depth=self.max_depth,
                node_type='right_node'
            )
            if right.gini_impurity == 0:
                right.node_type = "leaf"

            self.right = right
            self.right.grow_tree()

        else:
            self.node_type = "leaf"

    @staticmethod
    def calculate_best_divider(column, gini):
        """
        Calculating best divider for integer attribute
        If there is no good divider, returns None
        """
        #print(column)
        values = column[column.keys()[0]].values
        maximum, minimum = max(values), min(values)
        divider = minimum
        best_divider = divider
        result_values = (column[column.keys()[1]].unique())

        y1 = (column[column[column.keys()[1]] == result_values[0]]).shape[0]
        y2 = (column[column[column.keys()[1]] == result_values[1]]).shape[0]
        gini_base = Node.GINI_impurity(y1, y2)
        gini = 1
        while divider < maximum:
            divided_right = column[column[column.keys()[0]] >= divider]
            divided_left = column[column[column.keys()[0]] < divider]

            divided_right_count = divided_right.shape[0]
            divided_left_count = divided_left.shape[0]

            right_y1_count = (divided_right[divided_right[column.keys()[1]] == result_values[0]]).shape[
                0]
            right_y2_count = (divided_right[divided_right[column.keys()[1]] == result_values[1]]).shape[
                0]

            left_y1_count = (divided_left[divided_left[column.keys()[1]] == result_values[0]]).shape[
                0]
            left_y2_count = (divided_left[divided_left[column.keys()[1]] == result_values[1]]).shape[
                0]

            gini_right = Node.GINI_impurity(right_y1_count, right_y2_count)
            gini_left = Node.GINI_impurity(left_y1_count, left_y2_count)

            w_left = divided_left_count / (divided_left_count + divided_right_count)
            w_right = divided_right_count / (divided_left_count + divided_right_count)
            wGINI = w_left * gini_left + w_right * gini_right
            if wGINI < gini and w_left != 0 and w_right != 0:
                gini = wGINI
                best_divider = divider
            divider = divider + ((maximum - minimum) * 0.1)
        if gini_base <= gini or divided_right_count == 0 or divided_left_count == 0:
            return None

        return {
            "divider": best_divider,
            "gini": gini,
            "column": column.keys()[0]
        }

    @staticmethod
    def calculate_best_divider_str(column, gini):
        """
        Calculating best divider for string attribute
        If there is no good divider, returns None
        """
        values = column[column.keys()[0]].values
        values = np.unique(values)
        divider = values[0]
        best_divider = divider
        result_values = (column[column.keys()[1]].unique())

        y1 = (column[column[column.keys()[1]] == result_values[0]]).shape[0]
        y2 = (column[column[column.keys()[1]] == result_values[1]]).shape[0]
        gini_base = Node.GINI_impurity(y1, y2)


        gini = 1
        for v in values:
            divided_right = column[column[column.keys()[0]] == v]
            divided_left = column[column[column.keys()[0]] != v]

            divided_right_count = divided_right.shape[0]
            divided_left_count = divided_left.shape[0]

            right_y1_count = (divided_right[divided_right[column.keys()[1]] == result_values[0]]).shape[
                0]
            right_y2_count = (divided_right[divided_right[column.keys()[1]] == result_values[1]]).shape[
                0]

            left_y1_count = (divided_left[divided_left[column.keys()[1]] == result_values[0]]).shape[
                0]
            left_y2_count = (divided_left[divided_left[column.keys()[1]] == result_values[1]]).shape[
                0]

            gini_right = Node.GINI_impurity(right_y1_count, right_y2_count)
            gini_left = Node.GINI_impurity(left_y1_count, left_y2_count)

            w_left = divided_left_count / (divided_left_count + divided_right_count)
            w_right = divided_right_count / (divided_left_count + divided_right_count)
            wGINI = w_left * gini_left + w_right * gini_right
            if wGINI < gini and w_left != 0 and w_right != 0:
                gini = wGINI
                best_divider = v
        if gini_base <= gini or divided_right_count == 0 or divided_left_count == 0:
            return None
        return {
            "divider": best_divider,
            "gini": gini,
            "column": column.keys()[0]
        }

    @staticmethod
    def get_above_threshold_attributes(dataset, gini):
        """
        Returns all attributes with GINI above threshold
        threshold is GINI mean from all attributes
        """
        temp = []
        results = []
        i = 0
        column_number = len(dataset.axes[1])
        for column in dataset:
            if i == column_number - 1:
                break
            if (dataset[column].dtype == 'int64'):
                #print(dataset.iloc[:, [i, column_number - 1]])
                dictionary = Node.calculate_best_divider(dataset.iloc[:, [i, column_number - 1]], gini)

                if dictionary is None:
                    i = i + 1
                    continue

                results.append(dictionary)
            elif (dataset[column].dtype == 'object'):
                dictionary = Node.calculate_best_divider_str(dataset.iloc[:, [i, column_number - 1]], gini)
                #print(dataset.iloc[:, [i, column_number - 1]])

                if dictionary is None:
                    i = i + 1
                    continue
                results.append(dictionary)

            i = i + 1
        return results

    @staticmethod
    def get_random_attributes(dataset, gini):
        """
        Function returning 2 random attributes from dataset
        """

        # Getting attributes with GINI above threshold
        attributes = Node.get_above_threshold_attributes(dataset, gini)

        count = len(attributes)
        first_column_index = random.randint(0, count - 1)
        if count == 1:
            return attributes[first_column_index]
        second_column_index = random.randint(0, count - 1)
        while second_column_index == first_column_index:
            second_column_index = random.randint(0, count - 1)

        return [attributes[first_column_index], attributes[second_column_index]]

    @staticmethod
    def choose_best_column(dataset, gini):
        dictionaries = Node.get_random_attributes(dataset, gini)

        gini = 0
        result = None
        for dictionary in dictionaries:
            if dictionary['gini'] > gini:
                gini = dictionary['gini']
                result = dictionary

        return result
