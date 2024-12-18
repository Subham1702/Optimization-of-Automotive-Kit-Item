# Optimization-of-Automotive-Kit-Item

## Project Description: -
### Business Problem:
Automotive Manufacturers are facing difficulty in effectively sourcing and providing unique Kit Items (Automotive parts) from various vendors to meet customer demands.
### Business Objective: 
To maximize effective Kit-item delivery.
### Business Constraint: 
To maximize supply consistency.
### Business Success Criteria: 
To minimize the delayed delivery of kit items by at least 10%.
### Economic Success Criteria: 
To increase the revenue by at least 20 lacs INR.

## Data Understanding:
Data Dimension = 308 records, 42 attributes.

## Exploratory Data Analysis(EDA) & Data Preprocessing:

<details>
  <summary>EDA using MySQL</summary>

  ```SQL
CREATE DATABASE IF NOT EXISTS Automotive_db;
use Automotive_db;
drop table A_data;
create table if not exists A_data (
customer_code BIGINT NOT NULL,
customer_name VARCHAR(20) NOT NULL,
kit_item VARCHAR(50) NOT NULL,
OEM VARCHAR(50) NOT NULL,
item_desc VARCHAR(100) NOT NULL,
product_type VARCHAR(100) NOT NULL,
vehicle_1 VARCHAR(60) NOT NULL,
item_code VARCHAR(50) NOT NULL,
total_2021 INT NOT NULL,
total_2022 INT NOT NULL,
total_2023 INT NOT NULL,
grand_total INT NOT NULL
);
select * from A_data;

select count(distinct(kit_item)) from A_data;
select count(distinct(customer_name)) from A_data;
select count(distinct(OEM)) FROM A_data;

# most-frequent item #
SELECT kit_item AS mode_value, COUNT(*) AS frequency
FROM A_data GROUP BY kit_item
ORDER BY frequency DESC
LIMIT 1;

# least-frequent kit-item 
SELECT kit_item AS mode_value, COUNT(*) AS frequency
FROM A_data GROUP BY kit_item
ORDER BY frequency
LIMIT 1;

# check for null values
select count(*) from A_data where customer_code = '';
select count(*) from A_data where customer_name = '';
select count(*) from A_data where kit_item = '';
select count(*) from A_data where OEM = '';
select count(*) from A_data where item_desc = '';
select count(*) from A_data where product_type = '';
select count(*) from A_data where vehicle_1 = '';
select count(*) from A_data where item_code = '';
select count(*) from A_data where total_2021 = '';
select count(*) from A_data where total_2022 = '';
select count(*) from A_data where total_2023 = '';
select count(*) from A_data where grand_total = '';

                             -- 1st moment of business decision --
# Mean #
select avg(total_2021) from A_data;
select avg(total_2022) from A_data;
select avg(total_2023) from A_data;
   # Median #
WITH cte AS (
    SELECT total_2021,
           ROW_NUMBER() OVER (ORDER BY total_2021) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM A_data
)
SELECT AVG(total_2021) AS median_del
FROM cte
WHERE row_num IN (FLOOR((total_count + 1) / 2), CEIL((total_count + 1) / 2));

WITH cte AS (
    SELECT total_2022,
           ROW_NUMBER() OVER (ORDER BY total_2022) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM A_data
)
SELECT AVG(total_2022) AS median_del
FROM cte
WHERE row_num IN (FLOOR((total_count + 1) / 2), CEIL((total_count + 1) / 2));

WITH cte AS (
    SELECT total_2023,
           ROW_NUMBER() OVER (ORDER BY total_2023) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM A_data
)
SELECT AVG(total_2023) AS median_del
FROM cte
WHERE row_num IN (FLOOR((total_count + 1) / 2), CEIL((total_count + 1) / 2));

WITH cte AS (
    SELECT grand_total,
           ROW_NUMBER() OVER (ORDER BY grand_total) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM A_data
)
SELECT AVG(grand_total) AS median
FROM cte
WHERE row_num IN (FLOOR((total_count + 1) / 2), CEIL((total_count + 1) / 2));


  # Mode #
SELECT customer_code AS mode_value, COUNT(*) AS frequency FROM A_data GROUP BY customer_code ORDER BY frequency DESC LIMIT 1;  
SELECT customer_name AS mode_value, COUNT(*) AS frequency FROM A_data GROUP BY customer_name ORDER BY frequency DESC LIMIT 1;
SELECT kit_item AS mode_value, COUNT(*) AS frequency FROM A_data GROUP BY kit_item ORDER BY frequency DESC LIMIT 1;
SELECT OEM AS mode_value, COUNT(*) AS frequency FROM A_data GROUP BY OEM ORDER BY frequency DESC LIMIT 1;
SELECT item_desc AS mode_value, COUNT(*) AS frequency FROM A_data GROUP BY item_desc ORDER BY frequency DESC LIMIT 1;
SELECT product_type AS mode_value, COUNT(*) AS frequency FROM A_data GROUP BY product_type ORDER BY frequency DESC LIMIT 1;
SELECT vehicle_1 AS mode_value, COUNT(*) AS frequency FROM A_data GROUP BY vehicle_1 ORDER BY frequency DESC LIMIT 1;
SELECT item_code AS mode_value, COUNT(*) AS frequency FROM A_data GROUP BY item_code ORDER BY frequency DESC LIMIT 1;

select count(*) from A_data WHERE vehicle_1 = 'Truck';

						-- 2nd moment of business decision --
# range #
select round(MAX(total_2021) - MIN(total_2021), 4) as range_1 FROM A_data;
select round(MAX(total_2022) - MIN(total_2022), 4) as range_2 FROM A_data;
select round(MAX(total_2023) - MIN(total_2023), 4) as range_3 FROM A_data;
select round(MAX(grand_total) - MIN(grand_total), 4) as range_4 FROM A_data;

# variance #
select ROUND((variance(total_2021)), 4) as variance_1 from A_data;
select ROUND((variance(total_2022)), 4) as variance_2 from A_data;
select ROUND((variance(total_2023)), 4) as variance_3 from A_data;
select ROUND((variance(grand_total)), 4) as variance_4 from A_data;

# standard deviation #
select round((stddev(total_2021)), 4) FROM A_data;
select round((stddev(total_2022)), 4) FROM A_data;
select round((stddev(total_2023)), 4) FROM A_data;
select round((stddev(grand_total)), 4) FROM A_data;

						-- 3rd moment of business decision --
# Skewness #
SELECT
(
SUM(POWER(total_2021- (SELECT AVG(total_2021) FROM A_data), 3)) /
(COUNT(*) * POWER((SELECT STDDEV(total_2021) FROM A_data), 3))
) AS skewness FROM A_data;

SELECT
(
SUM(POWER(total_2022- (SELECT AVG(total_2022) FROM A_data), 3)) /
(COUNT(*) * POWER((SELECT STDDEV(total_2022) FROM A_data), 3))
) AS skewness FROM A_data;   

SELECT
(
SUM(POWER(total_2023- (SELECT AVG(total_2023) FROM A_data), 3)) /
(COUNT(*) * POWER((SELECT STDDEV(total_2023) FROM A_data), 3))
) AS skewness FROM A_data;    

SELECT
(
SUM(POWER(grand_total- (SELECT AVG(grand_total) FROM A_data), 3)) /
(COUNT(*) * POWER((SELECT STDDEV(grand_total) FROM A_data), 3))
) AS skewness FROM A_data;    

							-- 4th moment of business decision --
# Kurtosis #
SELECT
(
(SUM(POWER(total_2021- (SELECT AVG(total_2021) FROM A_data), 4)) /
(COUNT(*) * POWER((SELECT STDDEV(total_2021) FROM A_data), 4))) - 3
) AS kurtosis FROM A_data;

SELECT
(
(SUM(POWER(total_2021- (SELECT AVG(total_2021) FROM A_data), 4)) /
(COUNT(*) * POWER((SELECT STDDEV(total_2021) FROM A_data), 4))) - 3
) AS kurtosis FROM A_data;

SELECT
(
(SUM(POWER(total_2022- (SELECT AVG(total_2022) FROM A_data), 4)) /
(COUNT(*) * POWER((SELECT STDDEV(total_2022) FROM A_data), 4))) - 3
) AS kurtosis FROM A_data;

SELECT
(
(SUM(POWER(total_2023- (SELECT AVG(total_2023) FROM A_data), 4)) /
(COUNT(*) * POWER((SELECT STDDEV(total_2023) FROM A_data), 4))) - 3
) AS kurtosis FROM A_data;

SELECT
(
(SUM(POWER(grand_total- (SELECT AVG(grand_total) FROM A_data), 4)) /
(COUNT(*) * POWER((SELECT STDDEV(grand_total) FROM A_data), 4))) - 3
) AS kurtosis FROM A_data;


set sql_safe_updates = 0;
# imputation of missing values #
update A_data SET total_2021 = 0 WHERE total_2021 = '';
update A_data SET total_2022 = 0 WHERE total_2022 = '';
update A_data SET total_2023 = 0 WHERE total_2023 = '';

                        #-- Outlier Treatment --#
-- Inter-Quartile Range method --
-- Viewing the outlier values in Price column
WITH orderedList AS (
    SELECT total_2021, ROW_NUMBER() OVER (ORDER BY total_2021) AS row_n
    FROM A_data
),
iqr AS (
    SELECT
        total_2021,
        q3_value AS q_three,
        q1_value AS q_one,
        q3_value - q1_value AS outlier_range
    FROM orderedList
    JOIN (SELECT total_2021 AS q3_value FROM orderedList WHERE row_n = FLOOR((SELECT COUNT(*) FROM A_data) * 0.75)) q3 ON 1=1
    JOIN (SELECT total_2021 AS q1_value FROM orderedList WHERE row_n = FLOOR((SELECT COUNT(*) FROM A_data) * 0.25)) q1 ON 1=1
)
SELECT total_2021 AS outlier_value
FROM iqr
WHERE total_2021 >= q_three + outlier_range
   OR total_2021 <= q_one - outlier_range;
   
WITH orderedList AS (
    SELECT total_2022, ROW_NUMBER() OVER (ORDER BY total_2022) AS row_n
    FROM A_data
),
iqr AS (
    SELECT
        total_2022,
        q3_value AS q_three,
        q1_value AS q_one,
        q3_value - q1_value AS outlier_range
    FROM orderedList
    JOIN (SELECT total_2022 AS q3_value FROM orderedList WHERE row_n = FLOOR((SELECT COUNT(*) FROM A_data) * 0.75)) q3 ON 1=1
    JOIN (SELECT total_2022 AS q1_value FROM orderedList WHERE row_n = FLOOR((SELECT COUNT(*) FROM A_data) * 0.25)) q1 ON 1=1
)
SELECT total_2022 AS outlier_value
FROM iqr
WHERE total_2022 >= q_three + outlier_range
   OR total_2022 <= q_one - outlier_range;
   
WITH orderedList AS (
    SELECT total_2023, ROW_NUMBER() OVER (ORDER BY total_2023) AS row_n
    FROM A_data
),
iqr AS (
    SELECT
        total_2023,
        q3_value AS q_three,
        q1_value AS q_one,
        q3_value - q1_value AS outlier_range
    FROM orderedList
    JOIN (SELECT total_2023 AS q3_value FROM orderedList WHERE row_n = FLOOR((SELECT COUNT(*) FROM A_data) * 0.75)) q3 ON 1=1
    JOIN (SELECT total_2023 AS q1_value FROM orderedList WHERE row_n = FLOOR((SELECT COUNT(*) FROM A_data) * 0.25)) q1 ON 1=1
)
SELECT total_2023 AS outlier_value
FROM iqr
WHERE total_2023 >= q_three + outlier_range
   OR total_2023 <= q_one - outlier_range; 
   
                              
                                         ### DQL for initial insights ### 
select * from A_data WHERE vehicle_1 = 'Ciaz';
select count(distinct(vehicle_1)) from A_data;
select count(customer_name) from A_data WHERE vehicle_1 = 'Ciaz';

select * from A_data where vehicle_1 = 'Ciaz';
select * from A_data WHERE kit_item = 'KIT0001037' AND product_type = 'AC Unit';
select OEM from A_data WHERE kit_item = 'KIT0001037';


select * from c_data;

# number of unique customers
select count(distinct(cust_name)) from c_data;   # 76

# number of unique manufacturers
select count(distinct(OEM)) from c_data;       # 69

select * from c_data;
# number of unique KIT ITEM
select count(distinct(kit_item)) from c_data;    # 292
select count(distinct(vehicle)) from c_data;
select count(distinct(product_type)) from c_data;

select count(*) from c_data WHERE cust_name = 'Customer_62';
select count(*) from c_data WHERE cust_name = 'Customer_69';


                            
select distinct(vehicle) from c_data WHERE cust_name = 'Customer_69';  # Truck

# obtaining most frequent customer.
SELECT cust_name, COUNT(*) AS frequency
FROM c_data GROUP BY cust_name
ORDER BY frequency DESC LIMIT 1;      # Customer_39

select distinct(vehicle) from c_data WHERE cust_name = 'Customer_39';       # Truck
select distinct(OEM) from c_data WHERE cust_name = 'Customer_39';  # manufacturer 29, 48, 49
select distinct(product_type) from c_data WHERE cust_name = 'Customer_39';     # Exhaust Manifolds, Engine Fans, Radiators, Intercoolers
select count(distinct(kit_item)) from c_data WHERE cust_name = 'Customer_39';  # 23 items
 

select * from c_data WHERE OEM = 'manufacturer_29';
select distinct(cust_name) from c_data WHERE OEM = 'manufacturer_29';     # 4 customers
select distinct(item_desc) from c_data WHERE OEM = 'manufacturer_29';
select count(distinct(kit_item)) from c_data WHERE OEM = 'manufacturer_29';
select distinct(vehicle) from c_data WHERE OEM = 'manufacturer_29';     

select * from c_data WHERE kit_item = 'KIT0000560';  # connector_SUB
select * from c_data WHERE kit_item = 'KIT0000294';  # heater
select * from c_data WHERE kit_item = 'KIT0000414';   # child part
select * from c_data WHERE kit_item = 'KIT0001037';   # RS Evaporator

select distinct(item_desc) from c_data WHERE OEM = 'manufacturer_34';  # heater, child part, RS Evaporator, connector SUB
select distinct(vehicle) from c_data WHERE OEM = 'manufacturer_34';     # Ciaz

 # obtaining least frequent customer.
SELECT cust_name, COUNT(*) AS frequency
FROM c_data GROUP BY cust_name
ORDER BY frequency LIMIT 1;

select distinct(vehicle) from c_data WHERE cust_name = 'Customer_50';       # Truck
select distinct(OEM) from c_data WHERE cust_name = 'Customer_50';  # manufacturer 47
select distinct(product_type) from c_data WHERE cust_name = 'Customer_50';     # Electronics
select count(distinct(kit_item)) from c_data WHERE cust_name = 'Customer_50';  #1
```
</details>
<details>
  <summary>EDA using Python</summary>
	
  ```Python
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
  ```  
</details>

## Data Visualization: 

### Using Power BI
![alt text](https://github.com/Subham1702/Optimization-of-Automotive-Kit-Item/raw/main/Screenshot%20(329).png)

### Using Looker Studio
![alt text](https://github.com/Subham1702/Optimization-of-Automotive-Kit-Item/raw/main/Screenshot%20(330).png)

## Insights from the Data Analysis:
The dataset includes:
1) 76 customers.
2) 292 Kit-Items.
3) 69 Manufacturers.
### Statistical Insights: -
1) Demand and Sales trends over time.
```Python
plt.figure(figsize=(12, 6))
plt.plot(time_trend.index, time_trend['Total'], label='Total Ordered Quantity', linewidth=2)
plt.plot(time_trend.index, time_trend['parts_sold'], label='Parts Sold', linewidth=2)
plt.title('Demand and Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.show()	
```  
   ![alt text](https://github.com/Subham1702/Optimization-of-Automotive-Kit-Item/blob/main/Demand%20And%20Sales%20Trends%20Over%20Time.png)
   - Seasonal Demand and Inventory Optimization: Clear peaks and troughs in demand suggest seasonal or cyclical trends. Recognizing high-demand periods allows the company to optimize inventory and resource planning, preparing for potential order surges.
   - Supply Gaps and Fulfillment Improvement: Periods where sales fall short of demand indicate supply chain bottlenecks, missed sales opportunities, or unmet demand. Addressing these gaps through better forecasting and supplier coordination could enhance sales consistency and customer satisfaction.

2) Top Kit Items With Highest Average Delay.
```Python
top_kit_items_delay = kit_item_delay_trends_sorted.head(10)
plt.figure(figsize=(12, 6))
plt.barh(top_kit_items_delay['KIT ITEM'], top_kit_items_delay['avg_delay'], color='skyblue')
plt.xlabel('Average Delay')
plt.ylabel('Kit Item')
plt.title('Top Kit Items with Highest Average Delay')
plt.gca().invert_yaxis()
plt.show()
```     
   ![alt text](https://github.com/Subham1702/Optimization-of-Automotive-Kit-Item/blob/main/Top%20Kit%20Items%20With%20Highest%20Average%20Delay.png)
   - High Delay Concentration in Specific Items: Certain kit items consistently show higher average delays, such as HVAC units and gear shift levers, indicating recurring challenges in sourcing or manufacturing these products.
   - Potential Bottlenecks in High-Demand Parts: The frequent delays in critical components, like engine and transmission parts, suggest these items could be key bottlenecks, impacting overall delivery efficiency and customer satisfaction if not addressed.

3) Top Manufacturers By Delay Frequency.
```Python
top_manufacturers_delay = manufacturer_performance.nlargest(10, 'delay_frequency')
plt.figure(figsize=(12, 6))
plt.barh(top_manufacturers_delay['OEM'], top_manufacturers_delay['delay_frequency'], color='salmon')
plt.xlabel('Delay Frequency')
plt.ylabel('Manufacturer')
plt.title('Top Manufacturers by Delay Frequency')
plt.gca().invert_yaxis()
plt.show()
``` 
   ![alt text](https://github.com/Subham1702/Optimization-of-Automotive-Kit-Item/blob/main/Top%20Manufacturers%20By%20Delay%20Frequency.png)
   - Consistent Delays Among Key Manufacturers: A few manufacturers show a high frequency of delays, indicating they may struggle with reliability or capacity issues, which could be impacting overall supply chain efficiency.
   - Targeted Supplier Improvement Opportunities: By addressing delays with these specific manufacturers, the company could significantly reduce overall delivery delays, leading to improved supply consistency and better alignment with demand.

### Business Insights: -
1) Supply Chain Gaps:
  - High Delay Frequency by Key Manufacturers: Certain manufacturers are frequently associated with delayed deliveries, especially for critical parts like Engine Parts and Transmission components. This could indicate issues with supplier reliability, capacity constraints, or logistics bottlenecks.
  - Kit Items with Consistent Delays: Specific kit items, including HVAC units, gear shift levers, and stabilizer assemblies, have high average delays. These may be harder to source or require longer lead times, impacting overall fulfillment.

2) Customer Demand and Fulfillment Trends:
  - Fluctuating Demand: There are noticeable peaks in demand for certain items. Preparing for these demand spikes by building buffer stock or strengthening vendor relationships could reduce fulfillment delays and improve delivery consistency.
  - Fulfillment Rate Variability: Some product types, especially those related to AC Units and Transmission systems, show lower fulfillment rates, indicating a mismatch between ordered and available stock.
    
3) Opportunities for Improvement:
  - Focus on High-Impact Vendors and Kit Items: Addressing delay issues with vendors that frequently fall behind on deliveries could improve supply chain efficiency. Collaborating with these suppliers to streamline their processes or even considering alternative suppliers could help.
  - Inventory Optimization for High-Demand Periods: Building a demand forecast model based on historical trends would help preemptively manage inventory. Strategic inventory planning can reduce stockouts and enhance delivery performance, aligning with the objective of reducing delayed deliveries by 10%.
    
