# Packages required :

- lifelines is an open-source library for data analysis.
- numpy is the fundamental package for scientific computing in python.
- pandas is what we'll use to manipulate our data.
- matplotlib is a plotting library.

# Datasets :

- lymphoma dataset

# Survival Estimates

- survival function:

𝑆(𝑡)=𝑃(𝑇>𝑡)
 
To illustrate the strengths of Kaplan Meier, we'll start with a naive estimator of the above survival function. To estimate this quantity, we'll divide the number of people who we know lived past time  𝑡  by the number of people who were not censored before  𝑡 .

Formally, let  𝑖  = 1, ...,  𝑛  be the cases, and let  𝑡𝑖  be the time when  𝑖  was censored or an event happened. Let  𝑒𝑖=1  if an event was observed for  𝑖  and 0 otherwise. Then let  𝑋𝑡={𝑖:𝑇𝑖>𝑡} , and let  𝑀𝑡={𝑖:𝑒𝑖=1 or 𝑇𝑖>𝑡} . The estimator you will compute will be:

𝑆̂ (𝑡)=|𝑋𝑡||𝑀𝑡|