import numpy
import pandas
import sklearn.ensemble
import sklearn.model_selection

def load_data(file_name):
    """Loads data from a CSV file."""
    data = pandas.read_csv(file_name)
    # Sets the first column name,
    # it is missing in the CSV data.
    columns = list(data)
    columns[0] = 'id'
    data.columns = columns
    return data

def fill_data(data):
    """Fills-in the missing values in place."""
    for column in ('MonthlyIncome', 'NumberOfDependents'):
        # We use a replacement value outside the original interval.
        data[column].fillna(-1, inplace=True)

def save_data(data, file_name):
    """Saves a Kaggle submission from processed data."""
    with open(file_name, 'w') as stream:
        stream.write('Id,Probability\n')
        for index, row in data.iterrows():
            line = str(int(row['id'])) + ',' + str(row['Probability'])
            stream.write(line + '\n')

def vectorize(data):
    """Creates input and output vectors to fit/predict a model."""
    columns = list(data)[2:]
    x = data.as_matrix(columns=columns)
    y = data['SeriousDlqin2yrs'].values
    return x, y

def predict(x_train, y_train, x_test, model):
    """Predicts the probability of serious delinquency."""
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    # Fixes the probabilities outside the [0, 1] interval.
    y_predict[y_predict < 0] = 0
    y_predict[y_predict > 1] = 1
    return y_predict

def auroc_score(x, y, model):
    """Estimates the area under ROC curve of a model."""
    # We use k-fold cross-validation and average the scores.
    kfold = sklearn.model_selection.KFold(n_splits=5)
    scores = []
    for train_index, test_index in kfold.split(x):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        score = sklearn.metrics.roc_auc_score(
            y_test, predict(x_train, y_train, x_test, model))
        scores.append(score)
    return numpy.mean(scores)


# Training dataset
training_data = load_data('in/cs-training.csv')
fill_data(training_data)
x_train, y_train = vectorize(training_data)

# Test dataset
test_data = load_data('in/cs-test.csv')
fill_data(test_data)
x_test, y_test = vectorize(test_data)

model = model = sklearn.ensemble.BaggingRegressor(
    base_estimator=sklearn.ensemble.GradientBoostingRegressor(
        max_depth=4, n_estimators=130),
    n_estimators=30)

# Estimates the model score
print(auroc_score(x_train, y_train, model))

# Creates a Kaggle submission
test_data['Probability'] = predict(x_train, y_train, x_test, model)
save_data(test_data, 'out/submission.csv')
