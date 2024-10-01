# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 08:52:45 2024

@author: santa
"""

'''Data Preprocessing:- Converting raw data into formatted data '''

############# Data Pre-processing ##############

################ Type casting #################
import pandas as pd

data = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\ethnic diversity.csv")
data.dtypes 

'''
EmpID is Integer - Python automatically identify the data types by interpreting the values. 
As the data for EmpID is numeric Python detects the values as int64.

From measurement levels prespective the EmpID is a Nominal data as it is an identity for each employee.

If we have to alter the data type which is defined by Python then we can use astype() function

Explicit And Implicit Type Casting can be done 
'''

r=int('2')    #  Explicit
type(r)

e=4/2         # Implicit
type(e)

help(data.astype)

# Convert 'int64' to 'str' (string) type. 
data.EmpID = data.EmpID.astype('str')
data.dtypes

data.Zip = data.Zip.astype('str')
data.dtypes

# For practice:
# Convert data types of columns from:
    
# 'float64' into 'int64' type. 
data.Salaries = data.Salaries.astype('int64')
data.dtypes

# int to float
data.age = data.age.astype('float32')
data.dtypes


##############################################
### Identify duplicate records in the data ###
import pandas as pd
data = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\mtcars_dup.csv")

# Duplicates in rows
help(data.duplicated)


duplicate = data.duplicated() # by default keep=first   # Returns Boolean Series denoting duplicate rows. 
duplicate

sum(duplicate)

# Parameters
duplicate = data.duplicated(keep = 'last')
duplicate

# 17  2
# 23  6
# 27  23
 
duplicate = data.duplicated(keep = False)
duplicate


# Removing Duplicates
data1 = data.drop_duplicates() # Returns DataFrame with duplicate rows removed. by default keep=First
data1
# Parameters
data1 = data.drop_duplicates(keep = 'last')

data1= data.drop_duplicates(keep = False)   
sum(data1)

# Duplicates in Columns
# We can use correlation coefficient values to identify columns which have duplicate information

import pandas as pd

cars = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\Cars.csv")

# Correlation coefficient

cars.corr()

''' 
    Rule of thumb says |r| > 0.85 is a strong relation
    Rule of thumb says |r| < 0.4 is a weak relation
    Range for correlation coefficient ranges from -1 to 1
    1 indicates a perfect positive correlation (as one variable increases, the other increases).
   -1 indicates a perfect negative correlation (as one variable increases, the other decreases).
    0 indicates no linear correlation between the variables '''


'''We can observe that the correlation value for HP and SP is 0.973 and VOL and WT is 0.999 
& hence we can ignore one of the variables in these pairs.
'''


'''HP vs. SP: 0.974 → Strong positive correlation, meaning higher horsepower is strongly associated with higher speed.'''
'''HP vs MPG is -0.7250, which indicates a strong negative relationship. As horsepower increases, fuel efficiency tends to decrease'''
'''SP and MPG: Negative correlation (-0.687), indicating that faster vehicles tend to be less fuel-efficient'''
'''VOL and WT: showing that volume and weight are almost perfectly correlated. Larger vehicles tend to be heavier.'''
'''VOL And Weight 0.999 means both have capturing same information then why to keep both column why not drop 1 of them'''
'''HP And Top Speed might be capturing same kind of info as its near to 1, drop one of them '''
################################################
############## Outlier Treatment ###############
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\ethnic diversity.csv")
df.dtypes

# Let's find outliers in Salaries
sns.boxplot(df.Salaries)

sns.boxplot(df.age)
# No outliers in age column

# Detection of outliers (find limits for salary based on IQR)
IQR = df['Salaries'].quantile(0.75) - df['Salaries'].quantile(0.25)

lower_limit = df['Salaries'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Salaries'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# Let's flag the outliers in the dataset
# df.salaries
# df['salaries']
# df["salaries"]
outliers_df = np.where(df.Salaries > upper_limit, True, np.where(df.Salaries < lower_limit, True, False))

# outliers data
df_out = df.loc[outliers_df,]

df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# Let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Salaries)

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
df['df_replaced'] = pd.DataFrame(np.where(df['Salaries'] > upper_limit, upper_limit, np.where(df['Salaries'] < lower_limit, lower_limit, df['Salaries'])))
sns.boxplot(df.df_replaced)


############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5, 
                          variables = ['Salaries'])

df_s = winsor_iqr.fit_transform(df[['Salaries']])

# Inspect the minimum caps and maximum caps
 # winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(df_s.Salaries)


# Define the model with Gaussian method
df
winsor_gaussian = Winsorizer(capping_method = 'gaussian', 
                             # choose IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 3,
                          variables = ['Salaries'])

df_t = winsor_gaussian.fit_transform(df[['Salaries']])
sns.boxplot(df_t.Salaries)


# Define the model with percentiles:
# Default values
# Right tail: 95th percentile
# Left tail: 5th percentile

winsor_percentile = Winsorizer(capping_method = 'quantiles',
                          tail = 'both', # cap left, right or both tails 
                          fold = 0.05, # limits will be the 5th and 95th percentiles
                          variables = ['Salaries'])

df_p = winsor_percentile.fit_transform(df[['Salaries']])
sns.boxplot(df_p.Salaries)



##############################################
#### zero variance and near zero variance ####
'''Zero variance can apply to only numerical variables.'''

'''Zero variance in numerical data means that all the values in the variable are identical. For example,
   if you have a column where every value is 5, the variance of this column is zero because there is no spread
   or variability in the data.'''
   
'''Zero variance in categorical data means that all the categories are the same. For example, if you have a
   column where every value is "Yes", there is no variability, so the variance is considered zero.'''   
   
'''Variance Only Calculated On Numerical Column not on categorical column'''
   
import pandas as pd

df = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\ethnic diversity.csv")

df.dtypes
# If the variance is low or close to zero, then a feature is approximately constant and will not improve the performance of the model.
# In that case, it should be removed. 

df[['Salaries','age']].var() # variance of numeric variables

df[['Salaries','age']].var() == 0
# df[['Salaries','age']].var(axis = 0) == 0
df.var() == 0 #Error because categorical column is dere

df['Salaries'].mean()



#############
# Discretization

import pandas as pd
data = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\ethnic diversity.csv")
data.head()
data.tail()
data.head(10)
data.tail(10)

data.info()

data.describe()

# Binarization
data['Salaries_new'] = pd.cut(data['Salaries'], 
                              bins = [min(data.Salaries), data.Salaries.mean(), max(data.Salaries)],
                              labels = ["Low", "High"])

data 

# Look out for the break up of the categories.
data.Salaries_new.value_counts()


''' We can observe that the total number of values are 309. This is because one of the value has become NA.
This happens as the cut function by default does not consider the lowest (min) value while discretizing the values.
To over come this issue we can use the parameter 'include_lowest' set to True.
'''

data['Salaries_new1'] = pd.cut(data['Salaries'], 
                              bins = [min(data.Salaries), data.Salaries.mean(), max(data.Salaries)], 
                              include_lowest = True,
                              labels = ["Low", "High"])

data.Salaries_new1.value_counts()

#########
import matplotlib.pyplot as plt

plt.bar(x = range(310), height = data.Salaries_new1)
plt.hist(data.Salaries_new1)
plt.boxplot(data.Salaries_new1)


# Discretization / Multiple bins
data['Salaries_multi'] = pd.cut(data['Salaries'], 
                              bins = [min(data.Salaries), 
                                      data.Salaries.quantile(0.25),
                                      data.Salaries.mean(),
                                      data.Salaries.quantile(0.75),
                                      max(data.Salaries)], 
                              include_lowest = True,
                              labels = ["P1", "P2", "P3", "P4"])

data.Salaries_multi.value_counts()

data.MaritalDesc.value_counts()




##################################################
################## Dummy Variables ###############
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

# Use the ethinc diversity dataset
df= pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\ethnic diversity.csv")

df.columns # column names
df.shape # will give u shape of the dataframe

df.dtypes
df.info()

# Drop emp_name column
df=df.drop(['Employee_Name', 'EmpID', 'Zip'], axis = 1) #here u have to save it from original dataframe

'''OR'''
'''inplace = True,This parameter controls whether the changes should be made to the original DataFrame or if a new modified DataFrame should be returned. '''

df.drop(['Employee_Name', 'EmpID', 'Zip'], axis = 1, inplace = True) #here directly it saved


# Create dummy variables
'''dummies  refer to dummy variables, which are typically used when working with categorical data'''

df_new = pd.get_dummies(df)
df

# df_new_1 = pd.get_dummies(df, drop_first = True)
# Created dummies for all categorical columns

##### One Hot Encoding works

df.columns

# df = df[['Salaries', 'age', 'Position', 'State', 'Sex',
#           'MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department', 'Race']]

df=df[['MaritalDesc']]
df=df[['Sex']]

# a = df['Salaries']
# b = df[['Salaries']]

df

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One-Hot Encoder
enc = OneHotEncoder() # initializing method

# enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 2:]).toarray())
enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, :]).toarray())

#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder


df= pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\ethnic diversity.csv")
# Creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
X = df.iloc[:, :9]
# y = df.iloc[:, 9]

X['Sex'] = labelencoder.fit_transform(X['Sex'])
X['MaritalDesc'] = labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc'] = labelencoder.fit_transform(X['CitizenDesc'])





#################### Missing Values - Imputation ###########################
import numpy as np  
import pandas as pd

# Load modified ethnic dataset
df = pd.read_csv(r'C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\modified ethnic.csv') # for doing modifications

# Check for count of NA's in each column
df.isna()
df.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (Salaries)
# Mode is used for discrete data (ex: Position, Sex, MaritalDesc)

# For Mean, Median, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Mean Imputer 
df["Salaries"].isna().sum()
df["Salaries"].mean()
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
df["Salaries"] = pd.DataFrame(mean_imputer.fit_transform(df[["Salaries"]]))
df["Salaries"].isna().sum()

# Median Imputer
df["age"].isna().sum()
df["age"].median()
median_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
df["age"] = pd.DataFrame(median_imputer.fit_transform(df[["age"]]))
df["age"].isna().sum()  # all records replaced by median 

df.isna().sum()

# Mode Imputer
df["Sex"].isna().sum()
mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
df["Sex"] = pd.DataFrame(mode_imputer.fit_transform(df[["Sex"]]))
df.isnull().sum()


# Random Imputer
from feature_engine.imputation import RandomSampleImputer

df = pd.read_csv(r'C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\modified ethnic.csv') # for doing modifications


random_imputer = RandomSampleImputer(['age'])
df["age"] = pd.DataFrame(random_imputer.fit_transform(df[["age"]]))
df["age"].isna().sum()  # all records replaced by median







# df["MaritalDesc"] = pd.DataFrame(mode_imputer.fit_transform(df[["MaritalDesc"]]))


# df.isnull().sum()  # all Sex, MaritalDesc records replaced by mode

# # Constant Value Imputer
# constant_imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 'F')
# # fill_value can be used for numeric or non-numeric values

# df["Sex"] = pd.DataFrame(constant_imputer.fit_transform(df[["Sex"]]))





#####################
# Normal Quantile-Quantile Plot

import pandas as pd

# Read data into Python
education = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\education.csv")

import scipy.stats as stats
import pylab

# Checking whether data is normally distributed
'''Distribution:-normal'''
'''pylab :-Python Plots'''

stats.probplot(education.gmat, dist = "norm", plot = pylab)

stats.probplot(education.workex, dist = "norm", plot = pylab)   # If Exponential Use Log Transformation

import numpy as np

# Transformation to make workex variable normal

stats.probplot(np.log(education.workex), dist = "norm", plot = pylab)
# stats.probplot(np.sqrt(education.workex), dist = "norm", plot = pylab)

# Import modules
import pandas as pd
from scipy import stats

# Plotting related modules
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

# Read data into Python
education = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\education.csv")

# Original data
prob = stats.probplot(education.workex, dist = stats.norm, plot = pylab)

# Transform training data & save lambda value

'''The lambda (λ) value in the Box-Cox transformation typically ranges between -5 and 5'''

fitted_data, fitted_lambda = stats.boxcox(education.workex)

# Transformed data
prob = stats.probplot(fitted_data, dist = stats.norm, plot = pylab)


# creating axes to draw plots
fig, ax = plt.subplots(1, 2)

# Plotting the original data (non-normal) and fitted data (normal)
sns.distplot(education.workex, hist = True, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Non-Normal", color = "green", ax = ax[0])

sns.distplot(fitted_data, hist = True, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Normal", color = "green", ax = ax[1])

# adding legends to the subplots
plt.legend(loc = "upper right")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

print(f"Lambda value used for Transformation: {fitted_lambda}")



# Yeo-Johnson Transform

'''
We can apply it to our dataset without scaling the data.
It supports zero values and negative values. It does not require the values for 
each input variable to be strictly positive. 

In Box-Cox transform the input variable has to be positive.
'''

# import modules
import pandas as pd
from scipy import stats

# Plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

# Read data into Python
education = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\education.csv")

# Original data
prob = stats.probplot(education.workex, dist = stats.norm, plot = pylab)

from feature_engine import transformation

# Set up the variable transformer
tf = transformation.YeoJohnsonTransformer(variables = 'workex')

edu_tf = tf.fit_transform(education)

# Transformed data
prob = stats.probplot(edu_tf.workex, dist = stats.norm, plot = pylab)





####################################################
######## Standardization and Normalization #########
'''For Scaling We Need Only Numerical Values'''
import pandas as pd
import numpy as np

data= pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\mtcars.csv")

'''Describe Function can be applied only to numerical column'''
a = data.describe()

### Standardization
from sklearn.preprocessing import StandardScaler

# Initialise the Scaler
scaler = StandardScaler()

# To scale data
type(data)
df = scaler.fit_transform(data)
type(df)

df.describe()  # numpy.ndarray object has no attribute 'describe
# Convert the array back to a dataframe

dataset = pd.DataFrame(df)
res = dataset.describe()

# Normalization
''' Alternatively we can use the below function'''
from sklearn.preprocessing import MinMaxScaler
minmaxscale = MinMaxScaler()

df_n = minmaxscale.fit_transform(data)
dataset1 = pd.DataFrame(df_n)

res1 = dataset1.describe()


### Normalization
## load dataset
ethnic = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\ethnic diversity.csv")
ethnic.columns
ethnic.drop(['Employee_Name', 'EmpID', 'Zip'], axis = 1, inplace = True)

a1 = ethnic.describe()

# Get dummies
ethnic = pd.get_dummies(ethnic1)

a2 = ethnic.describe()

r=ethnic.Salaries
e=ethnic['age']
## Normalization function - Custom Function
# Range converts to: 0 to 1

'''Salaries'''

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(r)
b = df_norm.describe()



'''Age'''

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
df_norm = norm_func(e)
b = df_norm.describe()



''' Alternatively we can use the below function'''
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\mtcars.csv")

minmaxscale = MinMaxScaler()

mtcars_minmax = minmaxscale.fit_transform(data)
df_mtcars = pd.DataFrame(mtcars_minmax)
minmax_res = df_mtcars.describe()


'''Robust Scaling
Scale features using statistics that are robust to outliers'''

from sklearn.preprocessing import RobustScaler
data = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\DataPreprocessing_datasets\mtcars.csv")

robust_model = RobustScaler() # Initializing

df_robo = robust_model.fit_transform(data)

dataset_robust = pd.DataFrame(df_robo)
res_robust = dataset_robust.describe()










'''AUTO  EDA'''


# Load the Data
# Import the pandas library
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\santa\OneDrive\Desktop\360digitmg\module 7datasets\education.csv")


# Auto EDA
# ---------
# Sweetviz
# Autoviz
# Dtale
# Pandas Profiling
# Dataprep


# Sweetviz
###########
pip install sweetviz
# Import the sweetviz library
import sweetviz as sv

# Analyze the DataFrame and generate a report
s = sv.analyze(df)

# Display the report in HTML format
s.show_html()


































