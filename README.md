# Overview of the analysis
The objective at hand is to use machine learning to predict credit card risk. We test multiple algorithms for 
accuracy using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, to see 
which one best forecasts low and high-risk loan applications.

Each loan has about 85 features in the dataset. A couple of examples are Principal and Interest 
Received to Date, Most Recent Payment Amount, Interest Rate, Debt-to-Income Ratio, Months Since recent Credit Inquiry, and Home Ownership.

The issue with this dataset is that it has a significant bias in favour of good loans. Because most loans never default,
 99.9% of the loans in the database are considered low-risk. That's a lot of skewed data.

We utilize sklearn to split the data into training and testing sets to overcome the skewness of the data.
The testing data is then used to train models and make predictions.

The following criteria are used to assess the model's performance:

- Accuracy Score - This is just the percentage of right predictions, with 1 being 100% accurate and 0 being 0% accurate.

- Precision Score - a metric for how reliable a positive classification is, with 1 being 100% and 0 being 0%. As an example, "I'm aware that the high-risk test was positive. How  likely is it that the loan will be high-risk?"

- Recall Score - a measure of how many actual positives were accurately detected, with 1 being 100% correct and 0 being 0% correct. "I'm well aware that my loan is a high-risk investment. How likely is  that the test will be able to predict it?"

## Results

Let's take a look at six different machine learning models that can predict a high-risk loan application.

### Random Oversampling

- Random oversampling randomly selects instances of minority classes and adds them to the training set until the majority and minority classes are balanced.

- Accuracy: Expected to be 0.64 for high-risk use, which is actually correct.

![](/Resources/1_Randrom_Oversampling.png)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct. 

**Recall:** 0.61 of the actual high-risk application was correctly identified. 

![](/Resources/2_imbalanced_classification_report.png)

### SMOTE Oversampling

- The synthetic Minority Oversampling (SMOTE) technique increases the size of minorities by interpolating new instances. That is, some nearest neighbours are selected for an instance of the minority class. 

- Accuracy: Expected to be 0.62 for high-risk use, which is actually correct. 

![](/Resources/Smote_Score.png)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct.

**Recall:** 0.61 of the actual high-risk application was correctly identified. 

![](/Resources/3_Smote_oversampling.png)

### Cluster Centroids Undersampling

- Cluster Centroids identify majority class clusters and generate synthetic data points called centroids that represent the clusters. The majority class is then subsampled to the size of the minority class. 

- Accuracy: 0.52 for high-risk applications was predicted and was actually correct. 

![](/Resources/4_Cluster_Undersampling.png)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct.

**Recall:** 0.61 of the actual high-risk application was correctly identified. 

![](/Resources/4_Cluster_Report.png)
### SMOTEENN Combination Sampling

- The SMOTEENN is a combination of the SMOTE algorithm and the Edited Nearest  Neighbors (ENN) algorithm. SMOTEENN is a two-step process. 

*1.* Oversample  minority classes using SMOTE.

*2.* Use an undersampling strategy to clean up the resulting data. If the two nearest neighbors of a data point belong to two different classes, the data point will be deleted. 

- Accuracy: Expected high risk use of 0.639, actually correct.

![](/Resources/5_Smoteenn_sampling.png)

- Accuracy: A high-risk application of 0.01 is predicted and is actually correct.

Recall: 0.70 of the actual high-risk application was correctly identified.

![](/Resources/6_Smoteenn_classification.png)

### Random Forest Classifier

- The random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.

- Accuracy: 0.78 of high risk applications were predicted and actually correct.

![](/Resources/7_Balanced_Random_Forest.png)

- Precision: 0.03 of high risk applications were predicted and actually correct.

Recall: 0.70 of actual high risk applications identified correctly.

![](/Resources/8_Balanced_Report.png)

### Easy Ensemble Classifier

- Easy Ensemble selects all examples from the minority class and a subset from the majority class to create a balanced sample of the training set. Instead of using a pruned decision tree, a boosted decision tree is used for each subset, especially the AdaBoost algorithm.  

- Accuracy: 0.78 for high-risk use is expected and is actually correct.

![](/Resources/9_Easy_Acuracy_Score.png)








