import os
import zipfile
import logging
import warnings
import numpy as np
import pandas as pd
import configparser
import seaborn as sn
from sklearn import svm
from sklearn import tree
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def check_all_zip_files(path):
    zip_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.zip'):
                zip_files.append(os.path.join(root, file))
    return zip_files


def extract_zip(zip_file_path, destination_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)


def read_data(path, file_name):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == file_name:
                dataframe = pd.read_csv(os.path.join(root, file_name))
    return dataframe

