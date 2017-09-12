import matplotlib.pyplot
import numpy
import pandas

def create_bins(interval, bins_width):
    """Covers an interval with bins of the desired width."""
    bound = interval[0] - 0.5 * bins_width
    while bound < interval[1]:
        yield bound
        bound += bins_width
    yield bound

def create_axis(bins):
    """Creates an axis using the center of each bin."""
    i = 1
    while i < len(bins):
        yield 0.5 * (bins[i - 1] + bins[i])
        i += 1

def plot_compare(training_data, test_data, column, bins_width, interval=None):
    """Compares the distribution of a feature between the datasets."""
    # Creates the bins and the axis
    data = numpy.concatenate([
        training_data[column].values, test_data[column].values])
    if interval is None:
        interval = (min(data), max(data))
    bins = list(create_bins(interval, bins_width))
    axis = list(create_axis(bins))
    # Creates the histograms (training and test)
    training_histogram, _ = numpy.histogram(
        training_data[column].dropna().values, bins=bins)
    test_histogram, _ = numpy.histogram(
        test_data[column].dropna().values, bins=bins)
    # Plots and saves
    figure, subplots = matplotlib.pyplot.subplots(2, 1)
    subplots[0].set_title(column)
    subplots[0].bar(axis, training_histogram, bins_width, color='green')
    subplots[0].set_ylim(ymin=0)
    subplots[0].set_ylabel('training')
    subplots[1].bar(axis, test_histogram, bins_width, color='blue')
    subplots[1].set_ylim(ymin=0)
    subplots[1].set_ylabel('test')
    figure.tight_layout()
    figure.savefig('plots/compare-' + column + '.png')

def plot_analyze(data, column, bins_width, interval=None):
    """Analyzes the distribution of delinquents for a feature."""
    # Creates the bins and the axis
    if interval is None:
        interval = (min(data[column].values), max(data[column].values))
    bins = list(create_bins(interval, bins_width))
    axis = list(create_axis(bins))
    # Creates the histograms (non-delinquent and delinquent)
    timely_histogram, _ = numpy.histogram(
        data.query('SeriousDlqin2yrs == 0')[column].dropna().values, bins=bins)
    delinquent_histogram, _ = numpy.histogram(
        data.query('SeriousDlqin2yrs == 1')[column].dropna().values, bins=bins)
    # Derives the proportion of delinquents in each bin
    delinquent_proportion = []
    for i in range(len(axis)):
        timely_count = timely_histogram[i]
        delinquent_count = delinquent_histogram[i]
        if delinquent_count == 0: proportion = 0
        else: proportion = delinquent_count / (timely_count + delinquent_count)
        delinquent_proportion.append(proportion)
    # Plots and saves
    figure, subplots = matplotlib.pyplot.subplots(3, 1)
    subplots[0].set_title(column)
    subplots[0].bar(axis, timely_histogram, bins_width, color='blue')
    subplots[0].set_ylim(ymin=0)
    subplots[0].set_ylabel('non-delinquent')
    subplots[1].bar(axis, delinquent_histogram, bins_width, color='red')
    subplots[1].set_ylim(ymin=0)
    subplots[1].set_ylabel('delinquent')
    subplots[2].plot(axis, delinquent_proportion, color='black')
    subplots[2].set_ylim(ymin=0)
    subplots[2].set_ylabel('delinquent proportion')
    figure.tight_layout()
    figure.savefig('plots/analyze-' + column + '.png')


# Loads the datasets
training_data = pandas.read_csv('in/cs-training.csv')
test_data = pandas.read_csv('in/cs-test.csv')

# Compares the datasets
plot_compare(training_data, test_data, 'age', 1)
plot_compare(training_data, test_data, 'DebtRatio', 0.02, interval=(0, 1.2))
plot_compare(training_data, test_data, 'MonthlyIncome', 500, interval=(0, 25000))
plot_compare(training_data, test_data, 'NumberOfDependents', 1, interval=(0, 10))
plot_compare(training_data, test_data, 'NumberRealEstateLoansOrLines', 1, interval=(0, 15))
plot_compare(training_data, test_data, 'NumberOfOpenCreditLinesAndLoans', 1, interval=(0, 40))
plot_compare(training_data, test_data, 'RevolvingUtilizationOfUnsecuredLines', 0.02, interval=(0, 1.5))
plot_compare(training_data, test_data, 'NumberOfTime30-59DaysPastDueNotWorse', 1, interval=(0, 15))
plot_compare(training_data, test_data, 'NumberOfTime60-89DaysPastDueNotWorse', 1, interval=(0, 15))
plot_compare(training_data, test_data, 'NumberOfTimes90DaysLate', 1, interval=(0, 15))

# Analyses the features
plot_analyze(training_data, 'age', 1)
plot_analyze(training_data, 'DebtRatio', 0.02, interval=(0, 1.2))
plot_analyze(training_data, 'MonthlyIncome', 500, interval=(0, 25000))
plot_analyze(training_data, 'NumberOfDependents', 1, interval=(0, 10))
plot_analyze(training_data, 'NumberRealEstateLoansOrLines', 1, interval=(0, 15))
plot_analyze(training_data, 'NumberOfOpenCreditLinesAndLoans', 1, interval=(0, 40))
plot_analyze(training_data, 'RevolvingUtilizationOfUnsecuredLines', 0.02, interval=(0, 1.5))
plot_analyze(training_data, 'NumberOfTime30-59DaysPastDueNotWorse', 1, interval=(0, 15))
plot_analyze(training_data, 'NumberOfTime60-89DaysPastDueNotWorse', 1, interval=(0, 15))
plot_analyze(training_data, 'NumberOfTimes90DaysLate', 1, interval=(0, 15))
