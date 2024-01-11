# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:07:44 2024

@author: Molly
"""

import json
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

df1 = pd.read_csv("Bengaluru_House_Data.csv")
print(df1.head())
print(df1.shape)
print(df1.columns)
print(df1.groupby("area_type")["area_type"].agg('count'))
# df.agg({'column1': 'sum', 'column2': 'mean', 'column3': 'max'})
# df.agg({'column1': ['sum', 'mean'], 'column2': ['min', 'max']})

# To keep the project simple I will delete some columns & which has lots of NULL values
df2 = df1.drop(["area_type", "society", "balcony", "availability"],
               axis="columns")  # axis=0 by default # inplace=True but no
print(df2.head())

# 1. Date cleaning before any others
print(df2.isnull().sum())
# If they are big, calculate the mean value and fill it with NULL
# If not, just drop these rows
df3 = df2.dropna()
print(df3.isnull().sum())
# Then we notice there's sth wrong with the size column
print(df3["size"].unique())  # return all non-repetitive value
# Create a new column to save bhk
# x represents the value of each col 1 by 1
df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ")[0]))
# Check for house that has more than 20 bhk
print(df3[df3.bhk > 20])  # df3["bhk"] = df3.bhk
# Check if the bhk mateches with total square number
print(df3.total_sqft.unique())  # And we found range besides single number
# For the range, we will take the mean value of them
# Define a function to convert the value into float, see if it works


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# The conditions in [] must be True to return the corresponding rows, and False will be filtered out.
print(df3[~df3["total_sqft"].apply(is_float)])
# as we can see from the output, the data is un-uniformed, unstructured, contains outliers, has data errors
# To handle that, every time I have this range, take the mean value;
# every time I have the Sq. meter(wired data), ignore them

# the mean function


def convert_sqft_to_num(x):
    tokens = x.split("-")  # must be a string
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


print(convert_sqft_to_num("1804 - 2273"))  # 2038.5
print(convert_sqft_to_num("2166"))  # 2166.0
print(convert_sqft_to_num("34.46Sq. Meter"))  # not return anything

### We are creating new dataframe at each stage in our data processing pipeline ###
df4 = df3.copy()
df4["total_sqft"] = df3["total_sqft"].apply(
    convert_sqft_to_num)  # successfully converted
print(df4.loc[30])  # double check, yes it's the mean value of the range

# 2. Feature engineering
df5 = df4.copy()
# create a new feature help us do outlier cleaning later
df5["price_per_sqft"] = df5["price"]*100000/df5["total_sqft"]

# after done this, examing the location column
len(df5.location.unique())  # 1304 which is big bec it will have too many features

# first we find out how many data pointa are available
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby("location")["location"].agg(
    "count").sort_values(ascending=False)  # 1293 in total
# And there're many locations that only have 1 data point
# So come up with threshold, any locations have less than 10 datapoints
# is called other locations
# 1052 out of 1293 has less than 10 datapoints
len(location_stats[location_stats <= 10])
location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(
    lambda x: "Other" if x in location_stats_less_than_10 else x)
len(df5.location.unique())  # 242
# convenient for later one-hot encoding

# Outlier detection
# Issue 1: sqft
# We set the threshold as 300 sqf
# proved that these are unusual value, and we choose to remove them
df5[df5.total_sqft/df5.bhk < 300].head()
df5.shape  # (13246, 7)
# these are clearly data error/outliers, so can be safely removed

df6 = df5[~(df5.total_sqft/df5.bhk < 300)]  # filter all data errors
df6.shape  # (12502, 7)

# Issue 2: price
# check that is very high/low
# when using boolean filter, except for the condition, pandas will copy the rest by default, so here we don't need to copy()
df6.price_per_sqft.describe()
# min: 267(very unlikely) max: 176470(is likely but for general model, we'd better remove the extreme case)
# mean: 6308.5

# we assume that all the normal/general price should be between mean value and one standard deviation
# so we're going to filter any data that goes beyond one standard deviation

# remove outlier in .price_per_sqft


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):  # key=index subdf=df6[value]
        m = np.mean(subdf.price_per_sqft)
        stan = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-stan))
                           & (subdf.price_per_sqft <= (m+stan))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df7 = remove_pps_outliers(df6)
df7.shape  # (10241, 7) # removed almost 2000 extreme data

# next check if 3-bd has higher price than 2-bd when they have same sqft
# mayby it's because the location they are in, like downtown or countryside
# we want to know how many cases they have in the dataset
# create a visualization using scatter plot


def plot_scatter_chart(df, location):  # drawing a scatter plot
    # on which it will plot 2 bedrooms
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]  # 3 bedrooms
    matplotlib.rcParams["figure.figsize"] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color="blue", label="2 BHK", s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker="+",
                color="green", label="3 BHK", s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()

# plot_scatter_chart(df7, "Rajaji Nagar") # lood at 1700, 2 are more expensive than 3 vertically
# and they are assumed to be outliers
# plot_scatter_chart(df7, "Hebbal") # try another one, also the same situation
# so I'm goint to do some clean up in this area to fix it


# Create statics
{
    '1': {
        "mean": 4000,
        "std": 2000,
        "count": 34
    },
    '2': {
        "mean": 4300,
        "std": 2300,
        "count": 22
    }
}
# Filter out 2 beds whose value is less than mean of 1 bed


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk-1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(
                    exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats["mean"])].index.values)
    return df.drop(exclude_indices, axis="index")


df8 = remove_bhk_outliers(df7)
df8.shape  # (7329, 7)
# plot_scatter_chart(df8, "Hebbal") # compare to the previous one, 2 do not mix up with 3
# now 3 beds has more value/higher price than 2
# although there's still one green on there, it's inevitable to print it and it's fine

# now the outliers have been removed
# plot a Histogram to see how many apartments I have
# matplotlib.rcParams["figure.figsize"] = (20,10)
# plt.hist(df8.price_per_sqft,rwidth=0.8)
# plt.xlabel("Price Per Square Feet")
# plt.ylabel("Count")
# The result is between 0 and 15000, I have majority of my properties
# as shown is a normal distribution

# let's explore the bathroom feature
df8.bath.unique()
df8[df8.bath > 10]
# and we assume again, that my manager says
# every time you have 2 more bath than your bed, remove them as outliers
# let's plot a Histogram here
# plt.hist(df8.bath,rwidth=0.8) # width of the bar
# plt.xlabel("Number of bathrooms")
# plt.ylabel("Count")
# most of the house have 2, 4, 6bath, and 16 fx is outlier

df8[df8.bath > df8.bhk+2]  # remove
df9 = df8[df8.bath < df8.bhk+2]
df9.shape  # (7251, 7)

# now my data looks alomost clean and neat, so now we gonna move to ML traning
# and we have to drop unnecessary col such as price_per_sqft and size
# because we already have bhk==size, and pps is just for outlier cleaning
df10 = df9.drop(["size", "price_per_sqft"], axis="columns")
df10.head(3)


# *****************************************************************************#

# ML
# for each of the location, it will create a new col (one-hot encoding)
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop("Other", axis="columns")], axis="columns")

# becuase we already have them as 0 and 1
df12 = df11.drop("location", axis="columns")
df12.head(3)
# so this shows how the pipeline has formed, each stage has a dataframe correspondingly

# first examine the shape
df12.shape  # (7251, 245)

# for the model training
X = df12.drop("price", axis="columns")
X.head()  # all independent variable

# for price
y = df12.price
y.head()

# '''try to divide the dataset into training and testing
# use training dataset to train model
# use testing dataset to evaluate performance '''###
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)
# 20% sample to test and 80% to be trained

#
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)  # once the model is trained
# evaluate the socre of the model to tell how good it is
lr_clf.score(X_test, y_test)
# 0.8691914452174351 = 86.9% which is good

# we are going to use K fold cross validation first

cv = ShuffleSplit(n_splits=5, test_size=0.2,
                  random_state=0)  # randomize the sample
# that each of the fold have equal distribution
print(cross_val_score(LinearRegression(), X, y, cv=cv))
# [0.85430675 0.84187647 0.84728412 0.85171729 0.87168018]

# GridSearchCV


def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []  # store in the list
    # shuffle ramdomly my sample
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'],
                          cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


print(find_best_model_using_gridsearchcv(X, y))
# winner: linear_regression

# so we are gonna use the first model with 86.9% efficiency to predict price using couples of sample


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


predict_price('1st Phase JP Nagar', 1000, 2, 2)  # try this specific location
# 82.81981031345501 as estimated price
predict_price('1st Phase JP Nagar', 1000, 3, 3)
# 81.13648221355439 Okay not good
predict_price('Indira Nagar', 1000, 2, 2)
# 179.37066882807494
predict_price('Indira Nagar', 1000, 3, 3)
# 177.68734072817432

# then export the model to a pickle file.
# and it will be used by our python-Flask aserver.

with open("banglore_home_prices_model.pickle", "wb") as f:
    pickle.dump(lr_clf, f)


# for making a product prediction, the columns information
# the way they are structure and the index of the list
# so export them into a JSON file
columns = {
    "data_columns": [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
