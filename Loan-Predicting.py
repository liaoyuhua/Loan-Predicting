import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
raw_train = pd.read_csv("C:/Learning/TianChi/Loan-Defualt-Predicting/Data/train.csv")
raw_test = pd.read_csv("C:/Learning/TianChi/Loan-Defualt-Predicting/Data/testA.csv")
raw_train.head()
raw_test.head()

# EDA
# 1 basic data exploration
# 1.1 dimension of data set
raw_train.shape  # (800000, 47)
raw_test.shape  # (200000, 46)

# 1.2 distribution of response variable
sns.countplot(x=raw_train['isDefault'], data=raw_train)


# 1.3 categorical or numerical


# 1.4 missing values analysis
def missing_values_analysis(dataset, ignoreCompleteVar):
    """
    This function does analysis of missing values and visualizes it.
    :param dataset: dataframe with missing values
    :param ignoreCompleteVar: [True/False] whether return information of complete feature
    :return: a dataframe of distribution of missing values
    """
    df_missing_values = pd.DataFrame()
    var_name = []
    n_missing_values = []
    ratio_missing_values = []
    nrow = dataset.shape[0]
    for col in dataset.columns:
        n = dataset[col].isnull().sum()
        if n == 0 and ignoreCompleteVar == True:
            continue
        else:
            var_name.append(col)
            n_missing_values.append(n)
            ratio_missing_values.append(n / nrow)
    y_pos = range(len(var_name))
    plt.bar(y_pos, ratio_missing_values)
    plt.xlabel('Feature')
    plt.ylabel("Ratio of missing values")
    plt.title('Missing Values Bar Plot')
    plt.xticks(y_pos, var_name, rotation=90)
    plt.show()
    df_missing_values['var_name'] = var_name
    df_missing_values['n_missing_values'] = n_missing_values
    df_missing_values['ratio_missing_values'] = ratio_missing_values
    return df_missing_values


missing_values_analysis(raw_train, ignoreCompleteVar=True)


# 1.5 correlation analysis
# def correlation_analysis():


# 1.6 analysis of outliers

# 1.6.1 numerical method

# 1.6.2 graphical method

# 1.6.3 algorithm's method(DBSCAN)
class dbscan():
    def __init__(self, df, epsilon=1, minPts=5):
        self.df = np.array(df)
        self.epsilon = epsilon
        self.minPts = minPts
        self.cluster_label = 0
        self.noise = 0

    def fit(self):
        "Fit the data"
        self.df = np.append(self.df, np.array([[-1] * len(blobs)]).reshape(-1, 1), axis=1)
        for x in range(len(self.df)):
            # if the point is not labled already then search for neighbors
            if self.df[x, 2] != -1:
                continue

            # find neighbors
            p = self.df[x, :2]
            neighbors = self.rangeQuery(p)

            # If less neighbors than min_points then label as noise and continue
            if len(neighbors) < self.min_points:
                self.df[x, 2] = self.noise
                continue

            # increment cluster label
            self.cluster_label += 1

            # set current row to new cluster label
            self.df[x, 2] = self.cluster_label

            # create seed set to hold all neighbors of cluster including the neighbors already found
            found_neighbors = neighbors

            # create Queue to fold all neighbors of cluster
            q = Queue()

            # add original neighbors
            for x in neighbors:
                q.put(x)
            # While isnt empty label new neighbors to cluster
            while q.empty() == False:

                current = q.get()

                # if cur_row labled noise then change to cluster label (border point)
                if self.df[current, 2] == 0:
                    self.df[current, 2] = self.cluster_label

                # If label is not -1(unclassified) then continue
                if self.df[current, 2] != -1:
                    continue

                # label the neighbor
                self.df[current, 2] = self.cluster_label

                # look for neightbors of cur_row
                point = self.df[current, :2]
                neighbors2 = self.rangeQuery(point)

                # if neighbors2 >= min_points then add those neighbors to seed_set
                if len(neighbors2) >= self.min_points:

                    for x in neighbors2:
                        if x not in found_neighbors:
                            q.put(x)
                            found_neighbors.append(x)

            def predict(self, x):
                "Return the predicted labels"

                preds = []

                for point in x:
                    neighbors = self.rangeQuery(point)
                    label = self.df[neighbors[0], 2]
                    preds.append(label)

                return preds

            def rangeQuery(self, x):
                """Query database against x and return all points that are <= epsilon"""

                neighbors = []

                for y in range(len(self.df)):
                    q = self.df[y, :2]
                    if self.dist(x, q) <= self.epsilon:
                        neighbors.append(y)

                return neighbors

            def dist(self, point1, point2):
                """Euclid distance function"""

                x1 = point1[0]
                x2 = point2[0]
                y1 = point1[1]
                y2 = point2[1]

                # create the points
                p1 = (x1 - x2) ** 2
                p2 = (y1 - y2) ** 2

                return np.sqrt(p1 + p2)
# Data Preprocessing


# Modelling

# Tuning
