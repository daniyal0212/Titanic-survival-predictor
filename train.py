import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
import xgboost as xgb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load the data
def load_data():
    train_data = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test_data = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
    return train_data, test_data

# Check for missing values and data types
def check_data(train_data, test_data):
    print("Train Data Info:")
    print(train_data.info())
    print("\nTrain Data Description:")
    print(train_data.describe())
    print("\nTest Data Info:")
    print(test_data.info())
    print("\nTest Data Description:")
    print(test_data.describe())

# Make data ready for modeling
def preprocess_data(train_data, test_data):
    # Extract Title BEFORE dropping Name
    for df in [train_data, test_data]:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(
            ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
        )
        df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    train_data['Title'] = train_data['Title'].map(title_map).fillna(4).astype(int)
    test_data['Title'] = test_data['Title'].map(title_map).fillna(4).astype(int)
    # drop unnecessary columns
    train_data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
    test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
    

    # Handle missing values
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median()).astype(int)
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median()).astype(int)
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

    # Convert categorical variables to numeric
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
    train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # creating new features
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
    train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
    test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)
    train_data['Pclass_Sex'] = train_data['Pclass'] * train_data['Sex']
    test_data['Pclass_Sex'] = test_data['Pclass'] * test_data['Sex']
    
    return train_data, test_data

def rfe_selection(train_data):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Pclass_Sex', 'Title']
    X = train_data[features]
    Y = train_data['Survived']
    
    model = RandomForestClassifier(random_state=42)
    rfe = RFE(model, n_features_to_select=6)
    rfe.fit(X, Y)
    
    selected = [f for f, s in zip(features, rfe.support_) if s]
    print("RFE Selected:", selected)
    
    return selected


# Checking which model is working the best
# logistic regression 
def train_logistic_regression(train_data):
    features = selected_features  # using all selected features for modeling
    X = train_data[features]
    Y = train_data['Survived']
    
    model = LogisticRegression()
    scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    
    print(f"LR Accuracy: {round(scores.mean(), 4)} (+/- {round(scores.std(), 4)})")
    
    model.fit(X, Y)  # final fit on full training data
    return model
# Decision tree
def train_decision_tree(train_data):
    features = selected_features  # using all selected features for modeling
    X = train_data[features]
    Y = train_data['Survived']
    
    model = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    print(f"DT Accuracy: {round(scores.mean(), 4)} (+/- {round(scores.std(), 4)})")
    
    model.fit(X, Y)
    return model

# Random Forest
def train_random_forest(train_data):
    features = selected_features  # using all selected features for modeling
    X = train_data[features]
    Y = train_data['Survived']
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    print(f"RF Accuracy: {round(scores.mean(), 4)} (+/- {round(scores.std(), 4)})")
    model.fit(X, Y)
    return model

# XGBoost
def train_xgboost(train_data):
    features = selected_features  # using all selected features for modeling
    X = train_data[features]
    Y = train_data['Survived']
    
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    print(f"XGB Accuracy: {round(scores.mean(), 4)} (+/- {round(scores.std(), 4)})")
    
    model.fit(X, Y)
    return model

# tuning random forest as it had the best combination of accuracy and stability
def tune_random_forest(train_data):
    features = selected_features  # using all selected features for modeling
    X = train_data[features]
    Y = train_data['Survived']
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, Y)
    
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best Accuracy: {round(grid_search.best_score_, 4)}")
    
    return grid_search.best_estimator_

# tuning xgboost as it had the best combination of accuracy and stability
def tune_xgboost(train_data):
    features = selected_features  # using all selected features for modeling
    X = train_data[features]
    Y = train_data['Survived']
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10)
    grid_search.fit(X, Y)
    
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best Accuracy: {round(grid_search.best_score_, 4)}")
    
    return grid_search.best_estimator_

    
# prediction function to generate submission file
def predict(model, test_data):
    features = selected_features  # using all selected features for modeling
    X_test = test_data[features]
    
    predictions = model.predict(X_test)
    
    output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
    output.to_csv(os.path.join(BASE_DIR, 'submission1.csv'), index=False)
    print("Saved to submission1.csv")



# execution of the pipeline
train_data, test_data = load_data()
check_data(train_data, test_data)
train_data, test_data = preprocess_data(train_data, test_data)
selected_features = rfe_selection(train_data)
lr_model = train_logistic_regression(train_data)
dt_model = train_decision_tree(train_data)
rf_model = train_random_forest(train_data)
xgb_model = train_xgboost(train_data)
tuned_rf_model = tune_random_forest(train_data)
tuned_xgb_model = tune_xgboost(train_data)

# comparing all the models and selecting the best one based on cross-validation accuracy
def compare_all_models(models_dict, train_data):
    features = selected_features
    X = train_data[features]
    Y = train_data['Survived']

    results = {}
    for name, model in models_dict.items():
        scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
        results[name] = scores.mean()
        print(f"{name}: {round(scores.mean(), 4)} (+/- {round(scores.std(), 4)})")

    best_name = max(results, key=results.get)
    print(f"\n Best model: {best_name} ({round(results[best_name], 4)})")
    return models_dict[best_name]

candidates = {
    'LogisticRegression': lr_model,
    'DecisionTree': dt_model,
    'RandomForest': rf_model,
    'XGBoost': xgb_model,
    'TunedRandomForest': tuned_rf_model,
    'TunedXGBoost': tuned_xgb_model,
}

best_model = compare_all_models(candidates, train_data)
predict(best_model, test_data)


