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

- Accuracy: Expected to be 0.645 for high-risk use, which is actually correct.

![](/Resources/1_Random_score.png)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct. 

**Recall:** 0.61 of the actual high-risk application was correctly identified. 

![](/Resources/1_Random_report.png)

### SMOTE Oversampling

- The synthetic Minority Oversampling (SMOTE) technique increases the size of minorities by interpolating new instances. That is, some nearest neighbours are selected for an instance of the minority class. 

- Accuracy: Expected to be 0.623 for high-risk use, which is actually correct. 

![](/Resources/2_Smote_score.png)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct.

**Recall:** 0.61 of the actual high-risk application was correctly identified. 
 
![](/Resources/2_Smote_report.png)

### Cluster Centroids Undersampling

- Cluster Centroids identify majority class clusters and generate synthetic data points called centroids that represent the clusters. The majority class is then subsampled to the size of the minority class. 

- Accuracy: 0.529 for high-risk applications was predicted and was actually correct. 

![](/Resources/3_Cluster_score.png)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct.

**Recall:** 0.61 of the actual high-risk application was correctly identified. 

![](/Resources/3_Cluster_report.png)

### SMOTEENN Combination Sampling

- The SMOTEENN is a combination of the SMOTE algorithm and the Edited Nearest  Neighbors (ENN) algorithm. SMOTEENN is a two-step process. 

*1.* Oversample  minority classes using SMOTE.

*2.* Use an undersampling strategy to clean up the resulting data. If the two nearest neighbors of a data point belong to two different classes, the data point will be deleted. 

- Accuracy: Expected high risk use of 0.639, actually correct.

![](/Resources/4_Smoteenn_report.png)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct.

**Recall:** 0.70 of the actual high-risk application was correctly identified. 

![](/Resources/4_Smoteenn_score.png)

### Random Forest Classifier

- The random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.

- Accuracy: 0.788 of high risk applications were predicted and actually correct.

![](/Resources/5_Balanced_score.png)

**Precision:** A high-risk application of 0.03 is predicted and is actually correct.

**Recall:** 0.70 of the actual high-risk application was correctly identified. 

![](/Resources/5_Balanced_report.png)

### Easy Ensemble Classifier

- Easy Ensemble selects all examples from the minority class and a subset from the majority class to create a balanced sample of the training set. Instead of using a pruned decision tree, a boosted decision tree is used for each subset, especially the AdaBoost algorithm.  

- Accuracy: 0.788 for high-risk use is expected and is actually correct.

![](/Resources/6_Easy_score.png)

**Precision:** A high-risk application of 0.03 is predicted and is actually correct.

**Recall:** 0.70 of the actual high-risk application was correctly identified.

![](/Resources/6_Easy_report.png)


## Summary

It is interesting to find out that some of the above machine learning models outperform others. Given that, further study is needed to identify machine learning models that are more successful at making predictions.

However, considering the multiple methods listed above, the Easy Ensemble model is recommended, because each of its scores reveals that it is most likely to identify and anticipate high-risk loan applications effectively.

It's important to highlight that, as shown in the Classification Reports, its F1 score of 0.16 is much higher than the other models. The F1 score is a weighted average of the true positive rate (recall) and precision, with 1.0 being the highest and 0.0 being the worst.

There is usually a trade-off between sensitivity and accuracy, and you need to strike a balance between them. A convenient way to think about  F1 scores is that if there is a significant imbalance between sensitivity and accuracy, the F1 score will be low. While 0.16 is low, it is up to 8 times more than other models. Therefore, from the above options, the most preferable is to anticipate high-risk loan applications.

 #

**Contact:**
**E-mail:** mciano@hotmail.co.uk

**LinkedIn:** https://www.linkedin.com/in/marciorciano/

#