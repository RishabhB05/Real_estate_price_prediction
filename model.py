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
