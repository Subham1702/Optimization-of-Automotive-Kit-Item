# -*- coding: utf-8 -*-
"""
Created on Sun May 19 02:19:11 2024

@author: mital
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

raw_data = pd.read_csv(r"C:\Users\mital\Documents\Project-2 (Optimization Automotive kit-item) files\Automotive_dataset.csv")
raw_data.describe
raw_data.info
# missing values #
raw_data.isna().sum()


            # 1st Moment of business decision #
data['Total'].mean()
data['parts_sold'].mean()

data['parts_sold'].median()
data['Total'].median()

data['Customer Code'].mode()
data['Customer Name'].mode()
data['KIT ITEM'].mode()
data['Item Description'].mode()
data['Product type'].mode()
data['Vehicle 1'].mode()
data['Item Code'].mode()


                 # 2nd moment of business decision #
data['parts_sold'].std()
data['Total'].std()

data['parts_sold'].var()
data['Total'].var()

r1 = data['parts_sold'].max() - data['parts_sold'].min()
r2 = data['Total'].max() - data['Total'].min()


                # 3rd moment of business decision #
data['parts_sold'].skew()
data['Total'].skew()


                # 4th moment of business decision #
data['parts_sold'].kurt()
data['Total'].kurt()



           # Auto-EDA #
pip install -q autoviz
from autoviz.AutoViz_Class import AutoViz_Class  
AV = AutoViz_Class()
a = AV.AutoViz(r"C:\Users\mital\Documents\Project-2 (Optimization Automotive kit-item) files\Automotive_dataset.csv", chart_format='html')  

import os
os.getcwd()


pip install -q sweetviz
import pandas as pd
import sweetviz as sv

data = pd.read_csv(r"C:\Users\mital\Documents\Project-2 (Optimization Automotive kit-item) files\Automotive_dataset.csv")

# generate and display sweetviz EDA report
report = sv.analyze(data)
report.show_html()


                 
import seaborn as sns
sns.boxplot(data.Total)
sns.boxplot(data.parts_sold)

sns.histplot(data['Total'], kde=True)
sns.histplot(data['parts_sold'], kde=True)
data.dtypes

duplicate_count = data.duplicated().sum()
print(f'Total duplicate rows: {duplicate_count}')


                    # outlier treatment #
IQR = data['parts_sold'].quantile(0.75) - data['parts_sold'].quantile(0.25)
lower_limit = data['parts_sold'].quantile(0.25) - 1.5*IQR
upper_limit = data['parts_sold'].quantile(0.75) + 1.5*IQR

# flagging the outliers #
outliers_df = np.where(data.parts_sold > upper_limit, True, np.where(data.parts_sold < lower_limit, True, False))


# Replacing the outlier values with the upper and lower limits #
data['parts_sold'] = pd.DataFrame(np.where(data['parts_sold'] > upper_limit, upper_limit, np.where(data['parts_sold'] < lower_limit, lower_limit, data['parts_sold'])))

sns.boxplot(data.parts_sold)


IQR = data['Total'].quantile(0.75) - data['Total'].quantile(0.25)
lower_limit = data['Total'].quantile(0.25) - 1.5*IQR
upper_limit = data['Total'].quantile(0.75) + 1.5*IQR

# flagging the outliers #
outliers_df = np.where(data.Total > upper_limit, True, np.where(data.Total < lower_limit, True, False))


# Replacing the outlier values with the upper and lower limits #
data['Total'] = pd.DataFrame(np.where(data['Total'] > upper_limit, upper_limit, np.where(data['Total'] < lower_limit, lower_limit, data['Total'])))

sns.boxplot(data.Total)




                            # Transformation #
import scipy.stats as stats                            
import pylab

# checking for normal distribution #
stats.probplot(data['parts_sold'], dist = 'norm', plot=pylab)
stats.probplot(data['Total'], dist = 'norm', plot = pylab)


#  Function Transformation
import numpy as np           
stats.probplot(np.log(data.Total), dist = 'norm', plot = pylab)
stats.probplot(np.log(data.parts_sold), dist = 'norm', plot = pylab)

# power transformation - Yeo-Johnson transformation #
from feature_engine import transformation
tf = transformation.YeoJohnsonTransformer(variables='Total')
data_tf = tf.fit_transform(data)

prob = stats.probplot(data_tf['Total'], dist = 'norm', plot = pylab)


tf = transformation.YeoJohnsonTransformer(variables = 'parts_sold')
data_tf = tf.fit_transform(data)

prob = stats.probplot(data_tf.parts_sold, dist='norm', plot=pylab)


                   # encoding of categorical data #
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(data.iloc[:, 1:7]).toarray())


                    #----- Normalisation -----#
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

data['Total_norm'] = norm_func(data['Total'])


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

data['parts_sold_N'] = norm_func(data['parts_sold'])


data.rename(columns={'Customer Code':'cust_code', 'Customer Name':'cust_name','KIT ITEM':'kit_item', 'Item Description':'item_desc', 'Product type':'product_type', 'Vehicle 1':'vehicle', 'Item Code':'item_code'}, inplace=True)


               # pushing the data into MySQL 
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://root:password@Localhost/Automotive_db')
data.to_sql('c_data', con=engine, if_exists='replace', index=False)


                        ## UNIVARIATE ANALYSIS ##
# Set up the visualisation style
sns.set(style="whitegrid")

import matplotlib.pyplot as plt

# Univariate Analysis for Numerical Variables
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Distribution of 'Total'
sns.histplot(data['Total'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Total')
axes[0].set_xlabel('Total')

# Distribution of 'parts_sold'
sns.histplot(data['parts_sold'], bins=30, kde=True, ax=axes[1])
axes[1].set_title('Distribution of parts_sold')
axes[1].set_xlabel('parts_sold')

plt.tight_layout()
plt.show()




# Univariate Analysis for Categorical Variables
fig, axes = plt.subplots(3, 2, figsize=(18, 18))

# Define a function to plot count plots
def plot_countplot(variable, ax):
    sns.countplot(y=data[variable], order=data[variable].value_counts().index, ax=ax)
    ax.set_title(f'Frequency distribution of {variable}')

# Plotting the categorical variables
categorical_vars = ['Customer Name', 'KIT ITEM', 'OEM', 'Item Description', 'Product type', 'Vehicle 1']
for var, ax in zip(categorical_vars, axes.flatten()):
    plot_countplot(var, ax)

plt.tight_layout()
plt.show()




















