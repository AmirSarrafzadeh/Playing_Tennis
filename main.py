from config import *

# pippo = df[(df['weather condition'] == 'SUNNY') & (df['temperature'] == 'MILD')]
'''
Author = Amir Sarrafzadeh Arasi
Date = 2023-06-18
'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define logger
    logging.basicConfig(filename='report.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Read the config file
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        train_data_name = config['config']['train_data']
        test_data_name = config['config']['test_data']
    except Exception as e:
        logging.error('Error in reading config file: {}'.format(e))

    # Read all zip files
    path = os.getcwd()
    try:
        files = check_all_zip_files(path)
    except Exception as e:
        logging.error('Error in reading zip files: {}'.format(e))

    # Extract all zip files
    try:
        for file in files:
            extract_zip(file, path)
    except Exception as e:
        logging.error('Error in extracting zip files: {}'.format(e))

    # Read csv files
    try:
        df_train = read_data(path, train_data_name)
        df_test = read_data(path, test_data_name)
    except Exception as e:
        logging.error('Error in reading csv files: {}'.format(e))

    # Show the shape of dataframes
    try:
        rows_of_df_train = df_train.shape[0]  # Get the number of rows
        columns_of_df_train = df_train.shape[1]  # Get the number of columns
        rows_of_df_test = df_test.shape[0]  # Get the number of rows
        columns_of_df_test = df_test.shape[1]  # Get the number of columns
        logging.info(f"Train Dataframe has {rows_of_df_train} rows")
        logging.info(f"Train Dataframe has {columns_of_df_train} columns")
        logging.info(f"Test Dataframe has {rows_of_df_test} rows")
        logging.info(f"Test Dataframe has {columns_of_df_test} columns")
    except Exception as e:
        logging.error('Error in reading csv files rows and columns: {}'.format(e))

    # Show the columns of dataframes
    try:
        column_df_train = list(df_train.columns)
        column_df_test = list(df_test.columns)
        logging.info(f"Columns of Train Dataframe are: {column_df_train}")
        logging.info(f"Columns of Test Dataframe are {column_df_test}")
    except Exception as e:
        logging.error('Error in reading dataframe columns: {}'.format(e))

    # Show the data types of dataframes
    try:
        train_column_types = df_train.dtypes.tolist()  # Convert the data types to list
        test_column_types = df_test.dtypes.tolist()  # Convert the data types to list
        logging.info(f"Types of the columns of the Train Dataframe are: {train_column_types}")
        logging.info(f"Types of the columns of the Test Dataframe are: {test_column_types}")
    except Exception as e:
        logging.error('Error in reading train and test dataframe columns types: {}'.format(e))

    # Show the unique values of dataframes
    try:
        for column in column_df_train:
            logging.info(f"Unique values of {column} in Train Dataframe are: {df_train[column].unique()}")
    except Exception as e:
        logging.error('Error in getting unique values of train dataframe: {}'.format(e))

    try:
        for column in column_df_test:
            logging.info(f"Unique values of {column} in Test Dataframe are: {df_test[column].unique()}")
    except Exception as e:
        logging.error('Error in getting unique values of test dataframe: {}'.format(e))

    # Show the number of unique values of dataframes
    try:
        for column in column_df_train:
            logging.info(f"Number of unique values of {column} in Train Dataframe are: {df_train[column].nunique()}")
    except Exception as e:
        logging.error('Error in getting number of unique values of train dataframe: {}'.format(e))

    try:
        for column in column_df_test:
            logging.info(f"Number of unique values of {column} in Test Dataframe are: {df_test[column].nunique()}")
    except Exception as e:
        logging.error('Error in getting number of unique values of test dataframe: {}'.format(e))

    # Show the number of null values of dataframes
    try:
        for column in column_df_train:
            logging.info(f"Number of null values of {column} in Train Dataframe are: {df_train[column].isnull().sum()}")
    except Exception as e:
        logging.error('Error in getting number of null values of train dataframe: {}'.format(e))

    try:
        for column in column_df_test:
            logging.info(f"Number of null values of {column} in Test Dataframe are: {df_test[column].isnull().sum()}")
    except Exception as e:
        logging.error('Error in getting number of null values of test dataframe: {}'.format(e))

    # Show the number of unique values of dataframes
    try:
        for column in column_df_train:
            logging.info(f"Number of null values of {column} in Train Dataframe are: {df_train[column].value_counts()}")
    except Exception as e:
        logging.error('Error in getting number of unique values of train dataframe: {}'.format(e))

    try:
        for column in column_df_test:
            logging.info(f"Number of null values of {column} in Test Dataframe are: {df_test[column].value_counts()}")
    except Exception as e:
        logging.error('Error in getting number of unique values of test dataframe: {}'.format(e))

    ####################################################################################################################
    # Data Preprocessing
    # Before any Machine learning model, we need to preprocess the data. In this section, we will do the following:
    # Handle missing values
    # drop rows with more than one NaN value
    threshold = 2  # Number of NaN values allowed
    df_train = df_train.dropna(thresh=df_train.shape[1] - threshold + 1)

    # Show the rows of null values of dataframes
    df_nan = df_train[df_train.isnull().any(axis=1)]
    logging.info(f"Number of rows with null values in Train Dataframe are: {df_nan.shape[0]}")

    sample = df_train[(df_train['weather condition'] == 'SUNNY') & (df_train['temperature'] == 'MILD') & (
            df_train['windy'] == True) & (df_train['play'] == 'NO')]
    sample.dropna(inplace=True)
    df_train.loc[1, 'humidity'] = sample['humidity'].values[0]

    sample = df_train[
        (df_train['temperature'] == 'COOL') & (df_train['humidity'] == 'HIGHT') & (df_train['windy'] == True) & (
                df_train['play'] == 'NO')]
    sample.dropna(inplace=True)
    df_train.loc[4, 'weather condition'] = sample['weather condition'].values[0]

    sample = df_train[(df_train['weather condition'] == 'SUNNY') & (df_train['humidity'] == 'NORMAL') & (
            df_train['windy'] == False) & (df_train['play'] == 'YES')]
    sample.dropna(inplace=True)
    df_train.loc[7, 'temperature'] = sample['temperature'].values[0]

    sample = df_train[(df_train['weather condition'] == 'SUNNY') & (df_train['temperature'] == 'HOT') & (
            df_train['humidity'] == 'NORMAL') & (df_train['play'] == 'YES')]
    sample.dropna(inplace=True)
    df_train.loc[14, 'windy'] = sample['windy'].values[0]

    sample = df_train[
        (df_train['temperature'] == 'MILD') & (df_train['humidity'] == 'NORMAL') & (df_train['windy'] == False) & (
                df_train['play'] == 'NO')]
    sample.dropna(inplace=True)
    df_train.loc[15, 'weather condition'] = sample['weather condition'].values[0]

    sample = df_train[(df_train['weather condition'] == 'OVERCAST') & (df_train['temperature'] == 'HOT') & (
            df_train['windy'] == True) & (df_train['play'] == 'NO')]
    sample.dropna(inplace=True)
    df_train.loc[21, 'humidity'] = sample['humidity'].values[0]

    sample = df_train[(df_train['weather condition'] == 'OVERCAST') & (df_train['temperature'] == 'COOL') & (
            df_train['humidity'] == 'HIGHT') & (df_train['play'] == 'NO')]
    sample.dropna(inplace=True)
    df_train.loc[22, 'windy'] = sample['windy'].values[0]

    sample = df_train[(df_train['weather condition'] == 'OVERCAST') & (df_train['humidity'] == 'HIGHT') & (
            df_train['windy'] == True) & (df_train['play'] == 'NO')]
    sample.dropna(inplace=True)
    df_train.loc[28, 'temperature'] = sample['temperature'].values[0]

    sample = df_train[(df_train['weather condition'] == 'SUNNY') & (df_train['temperature'] == 'HOT') & (
            df_train['humidity'] == 'NORMAL') & (df_train['play'] == 'NO')]
    sample.dropna(inplace=True)
    df_train.loc[31, 'windy'] = sample['windy'].values[0]

    sample = df_train[(df_train['weather condition'] == 'SUNNY') & (df_train['humidity'] == 'NORMAL') & (
            df_train['windy'] == False) & (df_train['play'] == 'YES')]
    sample.dropna(inplace=True)
    df_train.loc[32, 'temperature'] = sample['temperature'].values[0]

    sample = df_train[(df_train['weather condition'] == 'SUNNY') & (df_train['humidity'] == 'NORMAL') & (
            df_train['windy'] == True) & (df_train['play'] == 'NO')]
    sample.dropna(inplace=True)
    df_train.loc[36, 'temperature'] = sample['temperature'].values[0]
    logging.info("ALL NaN values are filled with the appropriate values")

    # Drop deplicate rows
    df_train.drop_duplicates(inplace=True)
    logging.info("Duplicate rows are dropped from the Train Dataframe")

    df_test.drop_duplicates(inplace=True)
    logging.info("Duplicate rows are dropped from the Test Dataframe")

    # Encode categorical data
    string_to_int = preprocessing.LabelEncoder()  # encode your data
    df_train = df_train.apply(string_to_int.fit_transform)
    df_test = df_test.apply(string_to_int.fit_transform)

    df_test.rename(columns={'outlook': 'weather condition'}, inplace=True)
    ####################################################################################################################
    # Machine learning
    features = ['weather condition', 'temperature', 'humidity', 'windy']
    label = ['play']

    # Decision Tree Classifier
    X_train = df_train[features]
    y_train = df_train[label]
    X_test = df_test[features]
    y_test = df_test[label]

    classifier = DecisionTreeClassifier(criterion="entropy", random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy of decision tree classifier
    accuracy_dt = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of decision tree classifier is: {accuracy_dt}")

    # Calculate recall of decision tree classifier
    recall_dt = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of decision tree classifier is: {recall_dt}")

    # Calculate F1 score of decision tree classifier
    f1_dt = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of decision tree classifier is: {f1_dt}")

    # Data Visualization
    plt.figure(figsize=(15, 10))
    tree.plot_tree(classifier, filled=True)
    plt.savefig('decision_tree.png')
    plt.show()

    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Decision_tree.png')
    plt.show()

    # Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_rf = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Random Forest algorithm is: {accuracy_rf}")

    # Calculate recall
    recall_rf = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Random Forest algorithm is: {recall_rf}")

    # Calculate F1 score
    f1_rf = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Random Forest algorithm is: {f1_rf}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Random_Forest.png')
    plt.show()

    # Support Vector Machine Classifier (SVM)
    classifier = SVC(kernel='linear', random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_svm = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of SVM algorithm is: {accuracy_svm}")

    # Calculate recall
    recall_svm = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of SVM algorithm is: {recall_svm}")

    # Calculate F1 score
    f1_svm = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of SVM algorithm is: {f1_svm}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_SVM.png')
    plt.show()

    # Naive Bayes Classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_nb = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Naive Bayes algorithm is: {accuracy_nb}")

    # Calculate recall
    recall_nb = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Naive Bayes algorithm is: {recall_nb}")

    # Calculate F1 score
    f1_nb = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Naive Bayes algorithm is: {f1_nb}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Naive_Bayes.png')
    plt.show()

    # K-Nearest Neighbors Classifier (KNN)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_knn = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of KNN algorithm is: {accuracy_knn}")

    # Calculate recall
    recall_knn = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of KNN algorithm is: {recall_knn}")

    # Calculate F1 score
    f1_knn = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of KNN algorithm is: {f1_knn}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_KNN.png')
    plt.show()

    # Logistic Regression Classifier
    classifier = LogisticRegression(random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_lr = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Logistic Regression algorithm is: {accuracy_lr}")

    # Calculate recall
    recall_lr = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Logistic Regression algorithm is: {recall_lr}")

    # Calculate F1 score
    f1_lr = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Logistic Regression algorithm is: {f1_lr}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Logistic_Regression.png')
    plt.show()

    # Linear Discrimination Analysis (LDA)

    # Create the LDA model
    lda = LDA(n_components=1)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    # Decision Tree Classifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_dt_lda = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Decision Tree algorithm after applying LDA is: {accuracy_dt_lda}")

    # Calculate recall
    recall_dt_lda = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Decision Tree algorithm after applying LDA is: {recall_dt_lda}")

    # Calculate F1 score
    f1_dt_lda = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Decision Tree algorithm after applying LDA is: {f1_dt_lda}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Decision_Tree_LDA.png')
    plt.show()

    # Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_rf_lda = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Random Forest algorithm after applying LDA is: {accuracy_rf_lda}")

    # Calculate recall
    recall_rf_lda = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Random Forest algorithm after applying LDA is: {recall_rf_lda}")

    # Calculate F1 score
    f1_rf_lda = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Random Forest algorithm after applying LDA is: {f1_rf_lda}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Random_Forest_LDA.png')
    plt.show()

    # Support Vector Machine Classifier
    classifier = SVC(kernel='linear', random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_svm_lda = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of SVM algorithm after applying LDA is: {accuracy_svm_lda}")

    # Calculate recall
    recall_svm_lda = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of SVM algorithm after applying LDA is: {recall_svm_lda}")

    # Calculate F1 score
    f1_svm_lda = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of SVM algorithm after applying LDA is: {f1_svm_lda}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_SVM_LDA.png')
    plt.show()

    # Naive Bayes Classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_nb_lda = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Naive Bayes algorithm after applying LDA is: {accuracy_nb_lda}")

    # Calculate recall
    recall_nb_lda = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Naive Bayes algorithm after applying LDA is: {recall_nb_lda}")

    # Calculate F1 score
    f1_nb_lda = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Naive Bayes algorithm after applying LDA is: {f1_nb_lda}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Naive_Bayes_LDA.png')
    plt.show()

    # K-Nearest Neighbors Classifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_knn_lda = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of K-Nearest Neighbors algorithm after applying LDA is: {accuracy_knn_lda}")

    # Calculate recall
    recall_knn_lda = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of K-Nearest Neighbors algorithm after applying LDA is: {recall_knn_lda}")

    # Calculate F1 score
    f1_knn_lda = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of K-Nearest Neighbors algorithm after applying LDA is: {f1_knn_lda}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_KNN_LDA.png')
    plt.show()

    # Logistic Regression Classifier
    classifier = LogisticRegression(random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_lr_lda = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Logistic Regression algorithm after applying LDA is: {accuracy_lr_lda}")

    # Calculate recall
    recall_lr_lda = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Logistic Regression algorithm after applying LDA is: {recall_lr_lda}")

    # Calculate F1 score
    f1_lr_lda = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Logistic Regression algorithm after applying LDA is: {f1_lr_lda}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Logistic_Regression_LDA.png')
    plt.show()

    # Principal Component Analysis (PCA)
    pca = PCA(n_components=1)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Decision Tree Classifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_dt_pca = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Decision Tree algorithm after applying PCA is: {accuracy_dt_pca}")

    # Calculate recall
    recall_dt_pca = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Decision Tree algorithm after applying PCA is: {recall_dt_pca}")

    # Calculate F1 score
    f1_dt_pca = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Decision Tree algorithm after applying PCA is: {f1_dt_pca}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Decision_Tree_PCA.png')
    plt.show()

    # Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_rf_pca = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Random Forest algorithm after applying PCA is: {accuracy_rf_pca}")

    # Calculate recall
    recall_rf_pca = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Random Forest algorithm after applying PCA is: {recall_rf_pca}")

    # Calculate F1 score
    f1_rf_pca = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Random Forest algorithm after applying PCA is: {f1_rf_pca}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Random_Forest_PCA.png')
    plt.show()

    # Support Vector Machine Classifier
    classifier = SVC(kernel='linear', random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_svm_pca = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Support Vector Machine algorithm after applying PCA is: {accuracy_svm_pca}")

    # Calculate recall
    recall_svm_pca = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Support Vector Machine algorithm after applying PCA is: {recall_svm_pca}")

    # Calculate F1 score
    f1_svm_pca = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Support Vector Machine algorithm after applying PCA is: {f1_svm_pca}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Support_Vector_Machine_PCA.png')
    plt.show()

    # Naive Bayes Classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_nb_pca = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Naive Bayes algorithm after applying PCA is: {accuracy_nb_pca}")

    # Calculate recall
    recall_nb_pca = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Naive Bayes algorithm after applying PCA is: {recall_nb_pca}")

    # Calculate F1 score
    f1_nb_pca = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Naive Bayes algorithm after applying PCA is: {f1_nb_pca}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Naive_Bayes_PCA.png')
    plt.show()

    # K-Nearest Neighbors Classifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_knn_pca = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of K-Nearest Neighbors algorithm after applying PCA is: {accuracy_knn_pca}")

    # Calculate recall
    recall_knn_pca = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of K-Nearest Neighbors algorithm after applying PCA is: {recall_knn_pca}")

    # Calculate F1 score
    f1_knn_pca = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of K-Nearest Neighbors algorithm after applying PCA is: {f1_knn_pca}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_K_Nearest_Neighbors_PCA.png')
    plt.show()

    # Logistic Regression Classifier
    classifier = LogisticRegression(random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # Predicting the Test set results

    # Calculate accuracy
    accuracy_lr_pca = round(accuracy_score(y_test, y_pred), 2)
    logging.info(f"Accuracy of Logistic Regression algorithm after applying PCA is: {accuracy_lr_pca}")

    # Calculate recall
    recall_lr_pca = round(recall_score(y_test, y_pred), 2)
    logging.info(f"Recall of Logistic Regression algorithm after applying PCA is: {recall_lr_pca}")

    # Calculate F1 score
    f1_lr_pca = round(f1_score(y_test, y_pred), 2)
    logging.info(f"F1 Score of Logistic Regression algorithm after applying PCA is: {f1_lr_pca}")

    # Data Visualization
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_Logistic_Regression_PCA.png')
    plt.show()

    # Save results to Excel file for comparison
    results = {'accuracy': [accuracy_dt, accuracy_rf, accuracy_svm, accuracy_nb, accuracy_knn, accuracy_lr,
                            accuracy_dt_lda, accuracy_rf_lda, accuracy_svm_lda, accuracy_nb_lda, accuracy_knn_lda,
                            accuracy_lr_lda,
                            accuracy_dt_pca, accuracy_rf_pca, accuracy_svm_pca, accuracy_nb_pca, accuracy_knn_pca,
                            accuracy_lr_pca],
               'recall': [recall_dt, recall_rf, recall_svm, recall_nb, recall_knn, recall_lr
                   , recall_dt_lda, recall_rf_lda, recall_svm_lda, recall_nb_lda, recall_knn_lda, recall_lr_lda,
                          recall_dt_pca, recall_rf_pca, recall_svm_pca, recall_nb_pca, recall_knn_pca, recall_lr_pca],
               'f1': [f1_dt, f1_rf, f1_svm, f1_nb, f1_knn, f1_lr
                   , f1_dt_lda, f1_rf_lda, f1_svm_lda, f1_nb_lda, f1_knn_lda, f1_lr_lda,
                      f1_dt_pca, f1_rf_pca, f1_svm_pca, f1_nb_pca, f1_knn_pca, f1_lr_pca]}
    df = pd.DataFrame(results,
                      index=['Decision Tree', 'Random Forest', 'SVM', 'Naive Bayes', 'KNN', 'Logistic Regression'
                          , 'Decision Tree with LDA', 'Random Forest with LDA', 'SVM with LDA', 'Naive Bayes with LDA',
                             'KNN with LDA', 'Logistic Regression with LDA',
                             'Decision Tree with PCA', 'Random Forest with PCA', 'SVM with PCA', 'Naive Bayes with PCA',
                             'KNN with PCA', 'Logistic Regression with PCA'])
    df.to_excel('results.xlsx', index=True)
    logging.info(f"Results saved to Excel file")

    # Plot the results
    colors = ['blue', 'orange', 'green']

    # Create the x-axis locations for the bars
    x = np.arange(len(df.index))
    bar_width = 1 / (len(df.columns) + 1)

    # Set the figure size
    plt.figure(figsize=(16, 9))

    # Plot the bars for each metric
    for i, col in enumerate(df.columns):
        plt.bar(x + i * bar_width, df[col], color=colors[i % len(colors)], width=bar_width)

    # Add labels and title
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Evaluation Plot')

    # Add legend
    plt.legend(df.columns)

    # Customize x-axis tick labels
    plt.xticks(x + 0.5, df.index, rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()
