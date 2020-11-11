#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: # ACCURACY
#
# UPDATED 5/3/2020
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def compare_probs(prob1, prob2, epsilon):    # Uncertain if this will be in the final utils.py, adding it here.
    return abs(prob1 - prob2) <= epsilon

def enforce_demographic_parity(categorical_results, epsilon):   # Kedaar Raghavendra Rao
    import utils as u
    demographic_parity_data = {}
    thresholds = {}

    max_total_acc = 0
    temp_thresh = {} # {race, [thresh]}
    max_thresh = {} # {race, [thresh]}
    max_thresh_pred = {} # {race, [thresholded_pred]}
    temp_thresh_pred = {} # {race, [thresholded_pred]}

    race_cases = categorical_results  #categorical_results contains [predicted value, actual label]

    for p in range(1, 100):
        prob = p/100
        temp_thresh_pred = {}
        temp_thresh = {}
        for race in race_cases:
            for thresh in range(1, 100):
                t = thresh/100
                x = {}
                x[race] = u.apply_threshold(race_cases[race], t) #thresholded_pred
                r_prob = u.get_num_predicted_positives(x[race]) / len(x[race]) # u.get_true_positive_rate(x[race])
                if (compare_probs(r_prob, prob, epsilon)):
                    temp_thresh[race] = t
                    temp_thresh_pred[race] = x[race]
                    break

        if len(temp_thresh_pred) == len(race_cases):
            total_accuracy = u.get_total_accuracy(temp_thresh_pred)
            if total_accuracy > max_total_acc:
                max_total_acc = total_accuracy
                max_thresh_pred = temp_thresh_pred
                max_thresh = temp_thresh

    demographic_parity_data = max_thresh_pred
    thresholds = max_thresh

    return demographic_parity_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):     # Rishabh Sharma
    import utils as u
    thresholds = {}
    equal_opportunity_data = {}

    max_total_acc = 0
    temp_thresh = {} # {race, [thresh]}
    max_thresh = {} # {race, [thresh]}
    max_thresh_pred = {} # {race, [thresholded_pred]}
    temp_thresh_pred = {} # {race, [thresholded_pred]}

    race_cases = categorical_results   # categorical_results contains [predicted value, actual label]

    for p in range(1, 100):
        prob = p/100
        temp_thresh_pred = {}
        temp_thresh = {}
        for race in race_cases:
            for thresh in range(1, 100):
                t = thresh/100
                x = {}
                x[race] = u.apply_threshold(race_cases[race], t) #thresholded_pred
                r_prob = u.get_true_positive_rate(x[race])
                if (compare_probs(r_prob, prob, epsilon)):
                    temp_thresh[race] = t
                    temp_thresh_pred[race] = x[race]
                    break

        if len(temp_thresh_pred) == len(race_cases):
            total_accuracy = u.get_total_accuracy(temp_thresh_pred)
            if total_accuracy> max_total_acc:
                max_total_acc = total_accuracy
                max_thresh_pred = temp_thresh_pred
                max_thresh = temp_thresh      
        
    equal_opportunity_data = max_thresh_pred
    thresholds = max_thresh
    
    return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):   # Shamus O'Connor
    import numpy as np
    import utils
    mp_data = {}
    thresholds = {}

    for race in categorical_results:
        subset = categorical_results[race]
 
        max_index = 0  # Index that returns max accuracy
        acc = []
        
        thresh_num = len(subset)  # Num of thresholds to try
        thresh = np.linspace(0,1,thresh_num)

        for t in thresh:
            proc_data = []  # Data of each race sample for thresholding
            proc_data = utils.apply_threshold(subset, t)
            acc.append(utils.get_num_correct(proc_data)/len(proc_data)) # ACCURACY
#             acc.append(utils.apply_financials(proc_data,True))                 # COST

        max_index = acc.index(max(acc))  
        thresholds[race] = thresh[max_index]
        
        mp_data[race] = utils.apply_threshold(subset, thresholds[race])
       
    return mp_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):      # Kedaar Raghavendra Rao
    import utils as u
    predictive_parity_data = {}
    thresholds = {}

    max_total_acc = 0
    temp_thresh = {} # {race, [thresh]}
    max_thresh = {} # {race, [thresh]}
    max_thresh_pred = {} # {race, [thresholded_pred]}
    temp_thresh_pred = {} # {race, [thresholded_pred]}

    race_cases = categorical_results #categorical_results contains [predicted value, actual label]

    for p in range(1, 100):
        prob = p/100
        temp_thresh_pred = {}
        temp_thresh = {}
        for race in race_cases:
            for thresh in range(1, 100):
                t = thresh/100
                x = {}
                x[race] = u.apply_threshold(race_cases[race], t) #thresholded_pred
                r_prob = u.get_positive_predictive_value(x[race]) # u.get_num_predicted_positives(x[race]) / len(x[race])
                if (compare_probs(r_prob, prob, epsilon)):
                    temp_thresh[race] = [t]
                    temp_thresh_pred[race] = x[race]
                    break

        if len(temp_thresh_pred) == len(race_cases):
            total_accuracy = u.get_total_accuracy(temp_thresh_pred)
            if total_accuracy > max_total_acc:
                max_total_acc = total_accuracy
                max_thresh_pred = temp_thresh_pred
                max_thresh = temp_thresh

    predictive_parity_data = max_thresh_pred
    thresholds = max_thresh

    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):      # Shamus O'Connor
    import numpy as np
    import utils
    single_threshold_data = {}
    thresholds = {}
    
    full_list = []
    max_index = 0  # Index that returns max accuracy
    acc = []
  
    univ_thresh = 0;
    thresh_num = 500  # Num of thresholds to try
    thresh = np.linspace(0,1,thresh_num)
    
    for group in categorical_results.keys():  
        full_list += categorical_results[group]

    for t in thresh:        
        proc_data = []  # Data of each race sample for thresholding
        proc_data = utils.apply_threshold(full_list, t)
        acc.append(utils.get_num_correct(proc_data)/len(proc_data)) # ACCURACY
#         acc.append(utils.apply_financials(proc_data))                 # COST

    max_index = acc.index(max(acc))  
    univ_thresh = thresh[max_index]
  
    for race in categorical_results:
        subset = categorical_results[race]
        thresholds[race] = univ_thresh
        single_threshold_data[race] = utils.apply_threshold(subset, thresholds[race])     
    
    return single_threshold_data, thresholds