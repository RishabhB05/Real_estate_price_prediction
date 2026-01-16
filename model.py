import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('bengaluru_house_prices.csv')
# dropping useless columns
df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')


# ----------------------Data cleaning------------------------
# this will tell which row has NA
print(df2.isnull().sum())

# dropping rows with NA
df3 = df2.dropna()
print(df3.isnull().sum())


# in the csv size: some are given as 4BHK , while some are given as 4 Bedroom 
# so we make either BHK or Bedroom or remove the text after space
# checking unique values in size column
print(df3['size'].unique())
df3['BHK'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

# again checking after creating BHK column
print(df3.head())


# to check for error lets say bhk more than 15
print(df3[df3.BHK>15])  

# ṭhere are some error where square feet is less than bhk is more 

# there are some properties which are given in range like 2100-2850
# so we will take average of both


# so this will check if the value is a valid integer and then convert it into float
# if its not a valid inteqer such as range then it will return false
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# now we can see all the squarefeet that has range and not a valid float
print(df3[~df3['total_sqft'].apply(is_float)].head(5))


# in this model we will ignore any invalid sqft values and take and convert range to average
def convert_sqft_to_num(x):
    tokens = x.split('-') #1230 - 1380 it will split at - into two tokens
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2 #we return the mean of both
    try:
        return float(x)  # if its a valid float we return the float value
    except:
        return None
    
print(convert_sqft_to_num('2100-2850'))  # testing the function
print(convert_sqft_to_num('2166'))       # testing the function


# creating a new dataframe with converted sqft values
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)

# to access any slnumberd row
print(df4.loc[30])  # to check if the conversion is done properly



#-------------------- cleaning completed-----------------------

# ----------------------Feature Engineering------------------------

df5 = df4.copy()
# creating price per sqft column
df5['price_per_sqft'] = df5['price']*100000 / df5['total_sqft']

# lets see how many locations we have
print(len(df5['location'].unique())) #approax 1300 which is big

# so location has only 1 or 2 data points so we can remove those location
df5.location = df5.location.apply(lambda x: x.strip()) # to remove extra spaces

# grouping locations in descending order of count
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)

# so any location having less than 10 data points we will consider it as other
location_stats_less_than_10 = location_stats[location_stats<=10]

# replacing locations having less than 10 data points with 'other'
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# now later when we convert it into column we will have only few columns as we put those in other

# ----------------------Feature Engineering completed------------------------


# ----------------------Outlier Removal------------------------
# suppose minimum sqft required for 1 bhk is 300 sqft
annomaly = df5[df5.total_sqft/df5.BHK < 300]
print(annomaly.head()) #it will show all the rows where sqft per bhk is less than 300 sqft

# removing those annomalies
df6 = df5[~(df5.total_sqft/df5.BHK < 300)]

# removing extreme outliers in price per sqft
# describe function will give count, mean, std, min, 25%, 50%, 75%, max
check_extreme = df6.price_per_sqft.describe()
print(check_extreme)

# now we will remove data points which are beyond one standard deviation
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)
print(df7.shape)
# ----------------------Outlier Removal completed------------------------
# ----------------------Model Building------------------------

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
# this will plot scatter chart for given location
plot_scatter_chart(df7, "Rajaji Nagar")

# there are some anomalies where 3 bhk is less price than 2 bhk for same sqft area
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values
                )
    return df.drop(exclude_indices, axis='index')


df8 = remove_bhk_outliers(df7)
print(df8.shape)

# now its better
plot_scatter_chart(df8, "Rajaji Nagar") 

# now we will plot histogram of price per sqft
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
# plt.show()

# we also notice there are bathroom more than bhk so we fix that
df9 = df8[df8.bath < df8.BHK + 2]

# we drop size and price_per_sqft column as we have extracted all the information from it
df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')


# ----------------------Model Building completed------------------------
# -----------------------------Creating Dummy Variables------------------------


# converting categorical variable location using one hot encoding
# hot encoding is used to convert categorical variable into numerical variable

dummies = pd.get_dummies(df10.location)

# here we drop one column to avoid dummy variable trap
# means that if we have n categories then n-1 columns are enough to represent those categories
# other means locations which we grouped earlier as other and we dont need that column
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis = 'column')


# now we can drop the location column as we have converted it into multiple columns
df12 = df11.drop('location', axis='columns')

# -----------------------------Creating Dummy Variables completed------------------------










# -----------------------------Model Training------------------------
x = df12.drop('price', axis='columns') #everything except price
y = df12.price #only price column

# training and testing split
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train, y_train)
reg.score(x_test, y_test)  # score function will give r^2 value which tells how good our model is


# validating the model
# okay so what this will do is it will split the data into 5 different parts and each time it will take 4 parts as training data and 1 part as testing data
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

# here we are using 5 splits with test size of 20%
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# this will give r^2 value for each split
cross_val_score(LinearRegression(), x, y, cv=cv)


# -----------------------------Model Training completed------------------------

# -----------------------------L1 and L2 Regularization------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# this function will find the best model using grid search cv
# it will try different algorithms and different parameters for those algorithms and will return the best model

def find_best_model_using_gridsearchcv(x, y):
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

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=5, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

print(find_best_model_using_gridsearchcv(x, y))
# we got that linear regression is the best model for this dataset

# -----------------------------L1 and L2 Regularization completed------------------------   



# -----------------------------PREDICTION ON NEW DATA------------------------

def predict_price(location, sqft, bath, bhk):
    # first we find the index of the location column
    loc_index = np.where(x.columns==location)[0][0]
  
#   this line creates a zero array of length equal to number of columns in df12
# the reason we do this is because our model was trained on all the columns including the dummy variables for locations
# so when we want to predict for a new data point we need to create an array of same length with all values as 0
# then we will set the values for sqft, bath, bhk and the location column to 1
    x = np.zeros(len(df12.columns))
    # x[0] is sqft how did we got it? because in df12 the first column is total_sqft
    # x[1] is bath
    # x[2] is bhk
    # rest are location dummy variables
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # here we set the value of the location column to 1
    if loc_index >= 0:
        x[loc_index] = 1

    return reg.predict([x])[0]


print(predict_price('1st Phase JP Nagar', 1000, 2, 2))


# -----------------------------PREDICTION ON NEW DATA completed------------------------ 






# exporting the model to a pickle file
# this file can be used in flask app to make predictions

import pickle
with open('banglore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(reg, f)
# we also need to save the columns file which will be used in flask app
# why do we save the columns file? because when we get the input from user we need to convert it into the same format as our model
# so we need to know the order of columns and the location columns
import json
columns = {
    'data_columns' : [col.lower() for col in df12.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
    