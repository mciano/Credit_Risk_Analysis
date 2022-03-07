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

- Expected to be 0.67 for high-risk use, which is actually correct.

![](/Resources/1_Randrom_Oversampling.png)

- Accuracy: A high-risk application of 0.01 is predicted and is actually correct. 

Recall: 0.70 of the actual high-risk application was correctly identified. 

![](/Resources/2_imbalanced_classification_report.png)

### SMOTE Oversampling

- The synthetic Minority Oversampling (SMOTE) technique increases the size of minorities by interpolating new instances. That is, some nearest neighbours are selected for an instance of the minority class. 

- Accuracy: Expected to be 0.66 for high-risk use, which is actually correct. 

Recall: The actual high-risk application 0.63 was correctly identified.

![](/Resources/3_Smote_oversampling.png)






