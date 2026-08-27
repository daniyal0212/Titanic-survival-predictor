# Titanic-survival-predictor

A machine learning pipeline that predicts passenger survival on the Titanic using the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic).

## Overview

This project preprocesses the Titanic dataset, engineers new features, selects the most relevant ones via Recursive Feature Elimination (RFE), and trains/compares multiple classification models to find the best performer — then generates a submission file.

## Features Engineered

- **Title** — extracted from passenger names (Mr, Mrs, Miss, Master, Rare)
- **FamilySize** — SibSp + Parch + 1
- **IsAlone** — whether the passenger was traveling alone
- **Pclass_Sex** — interaction between passenger class and sex

## Models Compared

- Logistic Regression
- Decision Tree
- Random Forest (default + tuned via `GridSearchCV`)
- XGBoost (default + tuned via `RandomizedSearchCV`)

The model with the best 5-fold cross-validation accuracy is selected automatically and used for final predictions.

## Project Structure

```
Titanic-survival-predictor/
├── train.py          # main pipeline script
├── train.csv          # training data
├── test.csv            # test data
└── README.md
```

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/daniyal0212/Titanic-survival-predictor.git
   cd Titanic-survival-predictor
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost
   ```

3. Run the script:
   ```bash
   python train.py
   ```

This will preprocess the data, train and compare all models, and output `submission1.csv` in the project directory with the predictions from the best-performing model.

## Output

`submission1.csv` — contains `PassengerId` and predicted `Survived` (0 = did not survive, 1 = survived), formatted for Kaggle submission.

## License

Free to use for learning and experimentation.
