# Court-system-prediction

Northpointe’s COMPAS system is a privately developed algorithm used in many United States
courts as a metric for potential recidivism. COMPAS has been criticized by journalists for not
accounting for the base recidivism rates between racial groups, which is significantly more likely
to falsely flag black defendants as committing a crime in the future [1]. It is unclear if
Northpointe actually incorporated these base rates into its algorithm. Northpointe does not
incorporate race as a data feature, although racial bias may implicitly be incorporated through
implicit biases and historical inequalities [1].
In light of the disparity introduced by COMPAS, it is evident that a more fair algorithm is
needed. As an NGO centered around the belief of fairness for all, we believe that the variance in
accuracy between groups needs to be accounted for. Postprocessing methods that account for
hidden classes, like race, may be used to reduce this variance in accuracy between groups while
maximizing overall accuracy and reducing operational costs incurred by incorrect predictions.

#### Machine learning models investigated

1. A linear support vector regressor
2. A feed forward neural network
3. A naïve Bayes classifier

#### 5 potential post-processing methods

1. Maximum profit / maximum accuracy
2. Single threshold
3. Predictive parity
4. Demographic parity
5. Equal Opportunity

The objective of this method was to optimize accuracy and reduce variation in TPR and TNR
between groups . The method our NGO chose based on testing a myriad of machine learning
optimization techniques and postprocessing methods is a Support Vector Machine (SVM)
driven algorithm post-processed for equal opportunity .
In order to enforce equal opportunity, the TPR values (and thus, FNR values) between the
different sensitive groups were constrained to be equal to each other with an epsilon value of
±1%. This implies that the training data samples have a roughly equal chance of being labeled a
recidivist across all groups, which implies reduced bias. A secondary optimization for overall
accuracy was also applied to the model to find the thresholds, constrained by equal TPR values,
that also satisfy maximum accuracy.

After choosing the algorithm, the data set was divided into testing and training data. The final
model was retrained using the training data and was implemented on the testing data. It was
found that the overall accuracy on the test data was 64.79% and the cost was $-143,434,930.

#### Results

<img src="https://github.com/rs278/Court-system-prediction/blob/master/docs/Capture.PNG" width="400" height="600">

[1] Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016, May 23). Machine Bias. Retrieved
from www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing.

