import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv('Bank-data.csv')
print(raw_data.head())
data = raw_data.copy()

data = data.drop(['Unnamed: 0'], axis = 1)
# Using the map function to change any 'yes' values to 1 and 'no'values to 0. 
data['y'] = data['y'].map({'yes':1, 'no':0})

print(data.describe())

y = data['y']
x1 = data['duration']

x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
# Get the regression summary
print(results_log.summary())

plt.scatter(x1,y,color = 'C0')
# Don't forget to label your axes!
plt.xlabel('Duration', fontsize = 20)
plt.ylabel('Subscription', fontsize = 20)
plt.show()

# To avoid writing them out every time, we save the names of the estimators of our model in a list. 
estimators=['interest_rate','credit','march','previous','duration']

X1_all = data[estimators]
y = data['y']

X_all = sm.add_constant(X1_all)
reg_logit = sm.Logit(y,X_all)
results_logit = reg_logit.fit()
print(results_logit.summary2())

def confusion_matrix(data,actual_values,model):
             
   #Predict the values using the Logit model
  pred_values = model.predict(data)
   # Specify the bins 
  bins=np.array([0,0.5,1])
  # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
  # if they are between 0.5 and 1, they will be considered 1
  cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
  # Calculate the accuracy
  accuracy = (cm[0,0]+cm[1,1])/cm.sum()
  # Return the confusion matrix and 
  return cm, accuracy

# We have to load data our model has never seen before.
raw_data2 = pd.read_csv('Bank-data-testing.csv')
data_test = raw_data2.copy()
# Removes the index column thata comes with the data
data_test = data_test.drop(['Unnamed: 0'], axis=1)

# Coverting the outcome variable into 1s and 0s again. 
data_test['y'] = data_test['y'].map({'yes':1, 'no':0})
print(data_test.head())

y_test = data_test['y']
# We already declared a list called 'estimators' that holds all relevant estimators for our model. 
X1_test = data_test[estimators]
X_test = sm.add_constant(X1_test)

print('Confusion matrix for the test data:\n',confusion_matrix(X_test, y_test, results_logit))

print('Confusion matrix for the old data:\n',confusion_matrix(X_all,y, results_logit))