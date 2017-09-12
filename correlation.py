import pandas

def correlation(data, method, caption):
    """Calculates the correlation coefficients between columns.
    Displays them in descending order of their absolute values."""
    columns = list(data)
    coefficients = data.astype(float).corr(method=method)
    results = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            coefficient = coefficients[columns[i]][columns[j]]
            results.append((
                abs(coefficient), coefficient,
                columns[i] + ' x ' + columns[j]))
    print('# ' + caption + ', ' + method)
    for result in reversed(sorted(results)):
        abs_coefficient, coefficient, columns_pair = result
        print (coefficient, columns_pair)


# Training dataset
training_data = pandas.read_csv('in/cs-training.csv')
training_data.drop(training_data.columns[0], axis=1, inplace=True)
for method in ('pearson', 'spearman'):
    correlation(training_data, method, 'training')

# Test dataset
test_data = pandas.read_csv('in/cs-test.csv')
test_data.drop(test_data.columns[0], axis=1, inplace=True)
for method in ('pearson', 'spearman'):
    correlation(test_data, method, 'test')
