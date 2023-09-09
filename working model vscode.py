#!/usr/bin/env python
# coding: utf-8

# # Imports & Configs

# In[1]:


#uncomment the next line to download all the modules used in this project
#it only needs to run once per system used
#%pip install numpy pandas seaborn matplotlib optuna scikit-learn xgboost catboost lightgbm sklearn_relief deap


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
import optuna
import sklearn_relief as sr
import deap
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
import itertools
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from tabulate import tabulate
import os
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)



# # Data Preprocessing & EDA

# In[3]:


for dirname, _, filenames in os.walk('.\kaggle\input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        if 'Train' in filename:
            train=pd.read_csv(os.path.join(dirname, filename))
        if 'Test' in filename:
            test=pd.read_csv(os.path.join(dirname, filename))
train


# In[4]:


train.info()


# In[5]:


train.head()


# In[6]:


train.describe()


# In[7]:


train.describe(include='object')


# ## Missing Data

# In[8]:


total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count/total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")


# No missing values

# ## Duplicates 

# In[9]:


if train.duplicated().sum():
    print(f"Number of duplicate rows: {train.duplicated().sum()}")
    print(f"they will be removed")
    train.drop_duplicates(keep='first', inplace=True)
else:
    print(f"There are no duplicates within the dataset")


# ## Outliers 

# In[10]:


# for col in df:
#     if col != 'class' and is_numeric_dtype(df[col]):
#         fig, ax = plt.subplots(2, 1, figsize=(12, 8))
#         g1 = sns.boxplot(x = df[col], ax=ax[0])
#         g2 = sns.scatterplot(data=df, x=df[col],y=df['class'], ax=ax[1])
#         plt.show()


# No outliers

# In[11]:


plt.figure(figsize=(40,30))
sns.heatmap(train.corr(numeric_only=True), annot=True)

# import plotly.express as px
# fig = px.imshow(df.corr(), text_auto=True, aspect="auto")
# fig.show()


# In[12]:


sns.countplot(x=train['class'])


# # Label Encoding

# In[13]:


def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

le(train)
le(test)


# In[14]:


train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)
train.head()


# # Feature selection

# In[15]:


X_train = train.drop(['class'], axis=1)
Y_train = train['class']


# In[16]:


rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i==True]

selected_features


# In[17]:


X_train = X_train[selected_features]


# # Split and scale data

# In[18]:


scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)


# # K Nearest Neighbors (KNN) classification model

# In[20]:


def objective(trial):
    n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16, log=False)
    classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy


# In[21]:


study_KNN = optuna.create_study(direction='maximize')
study_KNN.optimize(objective, n_trials=1)
print(study_KNN.best_trial)


# In[22]:


KNN_model = KNeighborsClassifier(n_neighbors=study_KNN.best_trial.params['KNN_n_neighbors'])
KNN_model.fit(x_train, y_train)

KNN_train, KNN_test = KNN_model.score(x_train, y_train), KNN_model.score(x_test, y_test)

print(f"Train Score: {KNN_train}")
print(f"Test Score: {KNN_test}")


# # Logistic Regression Model

# In[23]:


lg_model = LogisticRegression(random_state = 42)
lg_model.fit(x_train, y_train)


# In[24]:


lg_train, lg_test = lg_model.score(x_train , y_train), lg_model.score(x_test , y_test)

print(f"Training Score: {lg_train}")
print(f"Test Score: {lg_test}")


# # Decision Tree Classifier

# In[25]:


def objective(trial):
    dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32, log=False)
    dt_max_features = trial.suggest_int('dt_max_features', 2, 10, log=False)
    classifier_obj = DecisionTreeClassifier(max_features = dt_max_features, max_depth = dt_max_depth)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy


# In[26]:


study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective, n_trials=30)
print(study_dt.best_trial)


# In[27]:


dt = DecisionTreeClassifier(max_features = study_dt.best_trial.params['dt_max_features'], max_depth = study_dt.best_trial.params['dt_max_depth'])
dt.fit(x_train, y_train)

dt_train, dt_test = dt.score(x_train, y_train), dt.score(x_test, y_test)

print(f"Train Score: {dt_train}")
print(f"Test Score: {dt_test}")


# In[28]:


fig = plt.figure(figsize = (30,12))
tree.plot_tree(dt, filled=True);
plt.show()


# In[29]:


from matplotlib import pyplot as plt

def f_importance(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.title('feature importance for dt')
    plt.show()

# whatever your features are called
features_names = selected_features

# Specify your top n features you want to visualize.
# You can also discard the abs() function 
# if you are interested in negative contribution of features
f_importance(abs(dt.feature_importances_), features_names, top=7)


# # Random Forest Classifier

# In[30]:


def objective(trial):
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=False)
    rf_max_features = trial.suggest_int('rf_max_features', 2, 10, log=False)
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 3, 20, log=False)
    classifier_obj = RandomForestClassifier(max_features = rf_max_features, max_depth = rf_max_depth, n_estimators = rf_n_estimators)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy


# In[31]:


study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective, n_trials=30)
print(study_rf.best_trial)


# In[32]:


rf = RandomForestClassifier(max_features = study_rf.best_trial.params['rf_max_features'], max_depth = study_rf.best_trial.params['rf_max_depth'], n_estimators = study_rf.best_trial.params['rf_n_estimators'])
rf.fit(x_train, y_train)

rf_train, rf_test = rf.score(x_train, y_train), rf.score(x_test, y_test)

print(f"Train Score: {rf_train}")
print(f"Test Score: {rf_test}")


# In[33]:


from matplotlib import pyplot as plt

def f_importance(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.title('feature importance for dt')
    plt.show()

# whatever your features are called
features_names = selected_features

# Specify your top n features you want to visualize.
# You can also discard the abs() function 
# if you are interested in negative contribution of features
f_importance(abs(rf.feature_importances_), features_names, top=7)


# # SKLearn Gradient Boosting Model

# In[34]:


SKGB = GradientBoostingClassifier(random_state=42)
SKGB.fit(x_train, y_train)


# In[35]:


SKGB_train, SKGB_test = SKGB.score(x_train , y_train), SKGB.score(x_test , y_test)

print(f"Training Score: {SKGB_train}")
print(f"Test Score: {SKGB_test}")


# # XGBoost Gradient Boosting Model

# In[36]:


xgb_model = XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(x_train, y_train)


# In[37]:


xgb_train, xgb_test = xgb_model.score(x_train , y_train), xgb_model.score(x_test , y_test)

print(f"Training Score: {xgb_train}")
print(f"Test Score: {xgb_test}")


# # Light Gradient Boosting Model

# In[38]:


lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(x_train, y_train)


# In[39]:


lgb_train, lgb_test = lgb_model.score(x_train , y_train), lgb_model.score(x_test , y_test)

print(f"Training Score: {lgb_train}")
print(f"Test Score: {lgb_test}")


# # SKLearn AdaBoost Model

# In[40]:


ab_model = AdaBoostClassifier(random_state=42)


# In[41]:


ab_model.fit(x_train, y_train)


# In[42]:


ab_train, ab_test = ab_model.score(x_train , y_train), ab_model.score(x_test , y_test)

print(f"Training Score: {ab_train}")
print(f"Test Score: {ab_test}")


# # CatBoost Classifier Model

# In[43]:


cb_model = CatBoostClassifier(verbose=0)


# In[44]:


cb_model.fit(x_train, y_train)


# In[45]:


cb_train, cb_test = cb_model.score(x_train , y_train), cb_model.score(x_test , y_test)

print(f"Training Score: {cb_train}")
print(f"Test Score: {cb_test}")


# # Naive Baye Model

# In[46]:


BNB_model = BernoulliNB()
BNB_model.fit(x_train, y_train)


# In[47]:


BNB_train, BNB_test = BNB_model.score(x_train , y_train), BNB_model.score(x_test , y_test)

print(f"Training Score: {BNB_train}")
print(f"Test Score: {BNB_test}")


# # Voting Model 

# In[48]:


v_clf = VotingClassifier(estimators=[('KNeighborsClassifier', KNN_model), ("XGBClassifier", xgb_model), ("RandomForestClassifier", rf), ("DecisionTree", dt), ("XGBoost", xgb_model), ("LightGB", lgb_model), ("AdaBoost", ab_model), ("Catboost", cb_model)], voting = "hard")


# In[49]:


v_clf.fit(x_train, y_train)


# In[50]:


voting_train, voting_test = v_clf.score(x_train , y_train), v_clf.score(x_test , y_test)

print(f"Training Score: {voting_train}")
print(f"Test Score: {voting_test}")


# # Bagging classifier

# In[51]:


baggin = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0)
baggin.fit(x_train , y_train)
bag_train, bag_test = baggin.score(x_train , y_train), baggin.score(x_test , y_test)

print(f"Training Score: {bag_train}")
print(f"Test Score: {bag_test}")


# # RReliefF classifier

# In[52]:


#baggin = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0)
#baggin.fit(x_train , y_train)

r = sr.RReliefF(n_features = 20)
print(r.fit_transform(x_train,y_train))
rrff_train, rrff_test = r.score(x_train , y_train), r.score(x_test , y_test)

print(f"Training Score: {rrff_train}")
print(f"Test Score: {rrff_test}")


# In[53]:


data = [["*KNN", KNN_train, KNN_test], 
        ["*Logistic Regression", lg_train, lg_test],
        ["Decision Tree", dt_train, dt_test], 
        ["Random Forest", rf_train, rf_test], 
        ["GBM", SKGB_train, SKGB_test], 
        ["XGBM", xgb_train, xgb_test], 
        ["*Adaboost", ab_train, ab_test], 
        ["light GBM", lgb_train, lgb_test],
        ["CatBoost", cb_train, cb_test], 
        ["*Naive Baye Model", BNB_train, BNB_test], 
        ["*Voting", voting_train, voting_test],
        ["*Baggings", bag_train, bag_test]]

col_names = ["Model", "Train Score", "Test Score"]
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
print('SVM Model takes a bit of time to run')
print('please wait while it runs')


# # SVM Model

# In[54]:


def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'linearSVC'])
    c = trial.suggest_float('c', 0.02, 1.0, step=0.02)
    if kernel in ['linear', 'rbf']:
        classifier_obj = SVC(kernel=kernel, C=c).fit(x_train, y_train)
    elif kernel == 'linearSVC':
        classifier_obj = LinearSVC(C=c).fit(x_train, y_train)
    elif kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 10)
        classifier_obj = SVC(kernel=kernel, C=c, degree=degree).fit(x_train, y_train)
        
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy


# In[55]:


study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective, n_trials=3)
print(study_svm.best_trial)


# In[56]:


if study_svm.best_trial.params['kernel'] in ['linear', 'rbf']:
    SVM_model = SVC(kernel=study_svm.best_trial.params['kernel'], C=study_svm.best_trial.params['c'])
elif kernel == 'linearSVC':
    SVM_model = LinearSVC(C=study_svm.best_trial.params['c'])
elif kernel == 'poly':
    SVM_model = SVC(kernel=study_svm.best_trial.params['kernel'], C=study_svm.best_trial.params['c'], degree=study_svm.best_trial.params['degree'])

SVM_model.fit(x_train, y_train)


# In[76]:


SVM_train, SVM_test = SVM_model.score(x_train , y_train), SVM_model.score(x_test , y_test)


print(f"Training Score: {SVM_train}")
print(f"Test Score: {SVM_test}")


# # Summary

# In[78]:


data = [["*KNN", f"{KNN_train*100:.2f}%", f"{KNN_test*100:.2f}%"], 
        ["*Logistic Regression", f"{lg_train*100:.2f}%", f"{lg_test*100:.2f}%"],
        ["Decision Tree", f"{dt_train*100:.2f}%", f"{dt_test*100:.2f}%"], 
        ["*Random Forest", f"{rf_train*100:.2f}%", f"{rf_test*100:.2f}%"], 
        ["GBM", f"{SKGB_train*100:.2f}%", f"{SKGB_test*100:.2f}%"], 
        ["XGBM", f"{xgb_train*100:.2f}%", f"{xgb_test*100:.2f}%"], 
        ["*Adaboost", f"{ab_train*100:.2f}%", f"{ab_test*100:.2f}%"], 
        ["light GBM", f"{lgb_train*100:.2f}%", f"{lgb_test*100:.2f}%"],
        ["CatBoost", f"{cb_train*100:.2f}%", f"{cb_test*100:.2f}%"], 
        ["*Naive Baye Model", f"{BNB_train*100:.2f}%", f"{BNB_test*100:.2f}%"], 
        ["*Voting", f"{voting_train*100:.2f}%", f"{voting_test*100:.2f}%"],
        ["*Baggings", f"{bag_train*100:.2f}%", f"{bag_test*100:.2f}%"],
        ["*SVM", f"{SVM_train*100:.2f}%", f"{SVM_test*100:.2f}%"]]
data.sort()

col_names = ["Model", "Train Score", "Test Score"]
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))

