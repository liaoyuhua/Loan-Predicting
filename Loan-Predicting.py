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

# Data Preprocessing


# Modelling

# Tuning
