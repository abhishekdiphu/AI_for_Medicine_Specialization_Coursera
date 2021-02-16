# Packages required :

- lifelines is an open-source library for data analysis.
- numpy is the fundamental package for scientific computing in python.
- pandas is what we'll use to manipulate our data.
- matplotlib is a plotting library.

# Datasets :

- lymphoma dataset

# Survival Estimates

## Naive survival function:

𝑆(𝑡)=𝑃(𝑇>𝑡)
 
To illustrate the strengths of Kaplan Meier, we'll start with a naive estimator of the above survival function. To estimate this quantity, we'll divide the number of people who we know lived past time  𝑡  by the number of people who were not censored before  𝑡 .

Formally, let  𝑖  = 1, ...,  𝑛  be the cases, and let  𝑡𝑖  be the time when  𝑖  was censored or an event happened. Let  𝑒𝑖=1  if an event was observed for  𝑖  and 0 otherwise. Then let  𝑋𝑡={𝑖:𝑇𝑖>𝑡} , and let  𝑀𝑡={𝑖:𝑒𝑖=1 or 𝑇𝑖>𝑡} . The estimator  compute 

𝑆̂ (𝑡)=|𝑋𝑡||𝑀𝑡|

## Kaplan-Meier estimate:

𝑆(𝑡)=∏𝑡𝑖≤𝑡(1−𝑑𝑖𝑛𝑖)
 
where  𝑡𝑖  are the events observed in the dataset and  𝑑𝑖  is the number of deaths at time  𝑡𝑖  and  𝑛𝑖  is the number of people who we know have survived up to time  𝑡𝑖 .



# Subgroup Analysis: 

We see that along with Time and Censor, we have a column called Stage_group.

- A value of 1 in this column denotes a patient with stage III cancer
- A value of 2 denotes stage IV.
- We want to compare the survival functions of these two groups.

KaplanMeierFitter class from lifelines is being used, and plotted for comparison.




# Log-Rank Test : 

To say whether there is a statistical difference between the survival curves we can run the log-rank test. 

- lets assume if the two curves were the same.

- then what is the  probability that we could observe this data. 

- if this p_value is less than a threshold alpha value 0f 0.05 then , we could say , there is less than 5 % chance that , we could observe this data.

- can be also called as a significance test.

- computed a p-value using lifelines.statistics.logrank_test.