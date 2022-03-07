# Overview of the analysis
The objective at hand is to use machine learning to predict credit card risk. We test multiple algorithms for accuracy using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, to see which one best forecasts low and high-risk loan applications.
Each loan has about 85 features (aka variables) in the dataset. A couple of examples are Principal and Interest Received to Date, Most Recent Payment Amount, Interest Rate, Debt-to-Income Ratio, Months Since recent Credit Inquiry, and Home Ownership.
The issue with this dataset is that it has a significant bias in favour of good loans. Because most loans never default, 99.9% of the loans in the database are considered low-risk. That's a lot of skewed data.
We utilize sklearn to split the data into training and testing sets to overcome the skewness of the data. The testing data is then used to train models and make predictions.
The following criteria are used to assess the model's performance:
