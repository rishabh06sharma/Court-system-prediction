from sklearn import svm
from Preprocessing import preprocess
from Report_Results import *
import numpy as np
from utils import *

metrics = ["sex", "age", 'race', 'c_charge_degree', 'priors_count', 'c_charge_desc'] # age_cat
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics, recalculate=False, causal=False)

def SVM_classification(metrics):

    np.random.seed(42)
    SVR = svm.LinearSVR(C=1.0/float(len(test_data)), max_iter=5000) # 5600
    SVR.fit(training_data, training_labels)

    #data = np.concatenate((training_data, test_data))
    #labels = np.concatenate((training_labels, test_labels))

    training_predictions = SVR.predict(training_data) # data
    testing_predictions = SVR.predict(test_data)
    return training_data, test_data, training_predictions, testing_predictions, training_labels, categories, mappings # labels

#######################################################################################################################

training_data, test_data, training_predictions, testing_predictions, labels, categories, mappings = SVM_classification(metrics)


training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, testing_predictions, test_labels)


epsilon = 0.01 # we use epsilon value to be 0.01 for Equal Opportunity.
training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, epsilon)

for group in training_race_cases.keys():
    training_race_cases[group] = apply_threshold(training_race_cases[group], thresholds[group])

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("Accuracy on test data:")
print(get_total_accuracy(test_race_cases))
print("")

print("Cost on test data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")
