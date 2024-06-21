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
                        