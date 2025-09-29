'''
About project -

The Immigration and Nationality Act (INA) of the US permits foreign workers to come to the United States 
to work on either a temporary or permanent basis. The act also protects US workers against adverse impacts 
on working place and maintain requirements when they hire foreign workers to fill workforce shortages.
 The immigration programs are administered by the Office of Foreign Labor Certification (OFLC).


 1) Problem statement.

* OFLC gives job certification applications for employers seeking to bring foreign workers into the United 
States and grants certifications. As In last year the count of employees were huge so OFLC needs Machine 
learning models to shortlist visa applicants based on their previous data.

**In this project we are going to use the data given to build a Classification model:**

* This model is to check if Visa get approved or not based on the given dataset.
* This can be used to Recommend a suitable profile for the applicants for whom the visa should be certified 
or denied based on the certain criteria which influences the decision.

## 2) Data Collection.
* The Dataset is part of Office of Foreign Labor Certification (OFLC)
* The data consists of 25480 Rows and 12 Columns

https://www.kaggle.com/datasets/moro23/easyvisa-dataset
'''


import pandas as pd

data = pd.read_csv(r"Visadataset.csv")
df = data.copy()
print(df.head())


print("No of Rows of data : ",df.shape[0])
print("....................................................\n")

print("No of columns of data : ",df.shape[1])
print("....................................................\n")

print("Display summary statistics for a dataframe :",df.describe())
print("....................................................\n")

print("Check Null and Dtypes: ",df.info())
print("....................................................\n")


print("Duplicates Rows: ",df.duplicated())
print("....................................................\n")


print("Count of Duplicate Rows: ",df.duplicated().sum())
print("....................................................\n")


print("Null Counts: ",df.isnull().sum())
print("....................................................\n")

print("All Columns: ",df.columns.to_list())
print("....................................................\n")


print("Numeric columns: \n",data.select_dtypes(include ='number').columns.to_list())
print(".....................................")

print("Catagorical columns: \n",data.select_dtypes(include ='object').columns.to_list())
print(".....................................")

# Drop irrelevant columns
df.drop(columns="case_id", axis=1, inplace=True)
df.info()

# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


# proportion of count data on categorical columns
for col in categorical_features:
    print(df[col].value_counts(normalize=True) * 100)
    print('---------------------------')


    #Save clean data
    df.to_csv("clean_data.csv",index=False)
    print("clean data saved successfully")

  

    print(df.head())


    '''

      **Insights**
 - `case_id` have unique vlaues for each column which can be dropped as it it of no importance
 - `continent` column is highly biased towards asia. hence we can combine other categories to form a single category.
 - `unit_of_wage` seems to be an important column as most of them are yearly contracts.
    
    
    '''
  
    
