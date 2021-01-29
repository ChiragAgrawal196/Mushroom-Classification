import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, \
    chi2  # chi2 aka. chi square is used when working with 2 categorical columns.
from scipy import stats
from dataframe_column_identifier import DataFrameColumnIdentifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle

missing_values = ["n/a", "na", "--", "NONE", "None", "none", "NA", "N/A", 'inf', '-inf']
data = pd.read_csv('E:\chirag\Datasets\Mushroom classification\mushrooms.csv', na_values=missing_values)

data = data.rename(columns={'cap-shape': 'cap_shape', 'cap-surface': 'cap_surface', 'cap-color': 'cap_color',
                            'gill-attachment': 'gill_attachment', 'gill-spacing': 'gill_spacing',
                            'gill-size': 'gill_size', 'gill-color': 'gill_color', 'stalk-shape': 'stalk_shape',
                            'stalk-root': 'stalk_root', 'stalk-surface-above-ring':
                                'stalk_surface_above_ring', 'stalk-surface-below-ring': 'stalk_surface_below_ring',
                            'stalk-color-above-ring': 'stalk_color_above_ring',
                            'stalk-color-below-ring': 'stalk_color_below_ring', 'veil-type': 'veil_type',
                            'veil-color': 'veil_color', 'ring-number': 'ring_number', 'ring-type': 'ring_type',
                            'spore-print-color': 'spore_print_color', 'class': 'target'})


# n_splits = 1 because I want to divide data into train and test sets
split = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)
for train_index, test_index in split.split(data, data['target']):
    stratified_train_data = data.loc[train_index]
    stratified_test_data = data.loc[test_index]

print(stratified_train_data.shape)
print(stratified_test_data.shape)

stratified_test_data.drop(['target'], 1, inplace=True)

le = LabelEncoder()
stratified_train_data[['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
                       'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
                       'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
                       'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color',
                       'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']] = \
    stratified_train_data[
        ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
         'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
         'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
         'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color',
         'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']].apply(le.fit_transform)

stratified_train_data['target'] = stratified_train_data['target'].replace('p', 0)
stratified_train_data['target'] = stratified_train_data['target'].replace('e', 1)

print(data)


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


corr_features = correlation(data.iloc[:, :21], 0.7)
print("Number of features deleted :", len(set(corr_features)))
print("Features deleted are :", corr_features)
print('FEATURES CORRELATION TO TARGET VALUES :')
train_data_corr = stratified_train_data[stratified_train_data.columns[1:]].corr()['target'][:]
print(train_data_corr)
print("====================================")
print('DELETING FEATURES THAT ARE LESS CORRELATED TO TARGET VARIABLES BETWEEN -0.1 & 0.1')
train_data_corr.drop(train_data_corr[(train_data_corr.values > -0.1) & (train_data_corr.values < 0.1)].index,
                     inplace=True)
print(train_data_corr)
print("====================================")
print("PRINTING THE DELETED COLUMN NAMES")
new_train_data = stratified_train_data.columns[~stratified_train_data.columns.isin(train_data_corr.index)]
print(new_train_data)

# DELETING ALL THE UNWANTED COLUMNS AND ALSO DELETING THE 'veil-type' COLUMN AS IT IS USELESS FOR US
stratified_train_data.drop(['spore_print_color', 'veil_color', 'ring_type', 'cap_shape', 'cap_color', 'veil_type'], 1,
                           inplace=True)

X = stratified_train_data.drop('target', 1)
y = stratified_train_data['target']


dfci = DataFrameColumnIdentifier()
select_K_Best = SelectKBest(k=9, score_func=chi2)
selected = select_K_Best.fit_transform(X, y)
selected_features = select_K_Best.get_support(indices=True)
print(selected_features)
print(dfci.select_columns_KBest(X, selected_features))
array_columns = np.array(X.columns)
print(array_columns[1:2], array_columns[4:5], array_columns[5:6], array_columns[6:7], array_columns[8:9],
      array_columns[9:10], array_columns[10:11], array_columns[14:15], array_columns[15:16])
print(X.columns)
new_X = X[['bruises', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_root', 'stalk_surface_above_ring',
          'stalk_surface_below_ring', 'population', 'habitat']]
print(new_X)


def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    return vif


print(calc_vif(new_X))

sum_column = new_X['stalk_surface_above_ring'] + new_X['stalk_surface_below_ring']
new_X['stalk_surface'] = sum_column
new_X.drop(['stalk_surface_above_ring', 'stalk_surface_below_ring'], 1, inplace=True)


def box_cox_transformation(df):
    try:
        for column in df:
            if (df[column].skew() > 1.0 or df[column].skew() < -1.0).any():
                plt.figure(figsize=(15, 6))
                plt.subplot(1, 2, 1)
                df[column].hist()

                plt.subplot(1, 2, 2)
                stats.probplot(df[column], dist="norm", plot=plt)
                print(df[column].skew())

                df[column], params = stats.boxcox(df[column] + 1)

                plt.figure(figsize=(15, 6))
                plt.subplot(2, 2, 1)
                df[column].hist()

                plt.subplot(2, 2, 2)
                stats.probplot(df[column], dist="norm", plot=plt)
                print(data[column].skew())

                return box_cox_transformation
    except TypeError:
        print("")


column = ['bruises', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_root', 'population', 'habitat', 'stalk_surface']
box_cox_transformation(new_X)

X_train, X_test, y_train, y_test = tts(new_X, y, random_state=42, test_size=0.3)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    errors = abs(y_pred - y_test)
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('Recall Score = ', recall_score(y_test, y_pred))
    print('Precision Score = ', precision_score(y_test, y_pred))
    return evaluate


def train_auc_roc_curve(model, X_test, y_test, X_train, y_train):
    y_pred = model.predict(X_test)
    print("roc curve :", roc_curve(y_test, y_pred))
    base_fpr, base_tpr, base_threshold = roc_curve(y_train, model.predict(X_train))
    plt.plot([0, 1])
    plt.plot(base_fpr, base_tpr)
    print("auc score :", auc(base_fpr, base_tpr))
    return train_auc_roc_curve


def test_auc_ruc_curve(model, X_test, y_test):
    test_fpr, test_tpr, test_threshold = roc_curve(y_test, model.predict(X_test))
    test_auc = auc(test_fpr, test_tpr)
    print(test_auc)
    plt.plot([0, 1])
    plt.plot(test_fpr, test_tpr)
    return test_auc_ruc_curve


base_model = LogisticRegression(random_state=1)
base_model.fit(X_train, y_train)
evaluate(base_model, X_test, y_test)
train_auc_roc_curve(base_model, X_test, y_test, X_train, y_train)


default_decision_tree_model = DecisionTreeClassifier(random_state=42)
default_decision_tree_model.fit(X_train, y_train)
evaluate(default_decision_tree_model, X_test, y_test)
feature_importance = default_decision_tree_model.feature_importances_.reshape(1, -1)
print(feature_importance.T)
features = np.array([X.columns[0:]])
print(features.T)
# decision_tree_feature_importance = pd.DataFrame(np.hstack((features.T, feature_importance.T)),
#                                                 columns=['feature', 'importance'])
# decision_tree_feature_importance['importance'] = pd.to_numeric(decision_tree_feature_importance['importance'])
# decision_tree_feature_importance.sort_values(by='importance', ascending=False)

# # ====================================================================================
# # ====================================================================================
stratified_test_data.head()

le = LabelEncoder()
stratified_test_data[['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
                      'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
                      'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
                      'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color',
                      'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']] = stratified_test_data[
    ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
     'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
     'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
     'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color',
     'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']].apply(le.fit_transform)

# DELETING ALL THE UNWANTED COLUMNS AND ALSO DELETING THE 'veil-type' COLUMN AS IT IS USELESS FOR US
stratified_test_data.drop(['cap_shape', 'cap_color', 'veil_type', 'veil_color', 'ring_type', 'spore_print_color'], 1,
                          inplace=True)

stratified_test_data['stalk_surface'] = stratified_test_data['stalk_surface_above_ring'] + stratified_test_data[
    'stalk_surface_below_ring']

stratified_test_data.drop(
    ['stalk_color_above_ring', 'stalk_color_below_ring', 'stalk_surface_below_ring', 'stalk_surface_above_ring',
     'cap_surface', 'odor', 'gill_attachment', 'stalk_shape', 'ring_number'], 1, inplace=True)

print("Remaining Columns are \n: {}".format(stratified_test_data.columns))
# column = ['bruises', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_root',
#           'population', 'habitat', 'stalk_surface']
box_cox_transformation(stratified_test_data)


predict = [1, 2, 1, 1, 2, 1, 3, 0]
predict = np.array(predict)
predict = predict.reshape(-1, 8)
My_prediction = default_decision_tree_model.predict(predict)
print(My_prediction)


test_auc_ruc_curve(default_decision_tree_model, X_test, y_test)

pickle.dump(default_decision_tree_model, open('model.pkl', 'wb'))
