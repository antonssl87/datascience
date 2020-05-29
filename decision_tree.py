# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## Data exploration and Cleaning

# %% [code]
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style="ticks",color_codes=True)

# %% [code]
#Reading the file from the directory and creating a dataframe for exploration
iowa_file_path = '/kaggle/input/iowa-house-prices/train.csv'

housing_data = pd.read_csv(iowa_file_path)

# %% [markdown]
# #### 1. Check available features for data exploration

# %% [code]
housing_data.columns

# %% [code]


# %% [code]


# %% [markdown]
# #### 2. Checking the dimensions of the dataframe

# %% [code]
housing_data.shape

# %% [markdown]
# #### 3. Explore what the data looks like

# %% [code]
housing_data.head()

# %% [markdown]
# As you can see we have missing values/NaN

# %% [markdown]
# #### 4. Couting Missing Data for each feature

# %% [code]
housing_data.isna().sum()

# %% [markdown]
# #### 5. Select interesting features for exploration and create a separate dataframe

# %% [code]
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd','SalePrice']

# %% [code]
housing_data_cleaned =  housing_data[feature_names]
housing_data_cleaned

# %% [markdown]
# #### 6. Check if there are null values from the selected features

# %% [code]
housing_data_cleaned.isna().sum()

# %% [markdown]
# #### 7. Get a short insight for what is inside those features

# %% [code]
housing_data_cleaned.describe()

# %% [markdown]
# ## Visualize Data

# %% [markdown]
# ### Pair plot to check relationship between each feature

# %% [code]
plt.figure(dpi=30)
sns.pairplot(housing_data_cleaned)
plt.show()

# %% [markdown]
# ### Exploring distributions of each feature

# %% [markdown]
# #### 1. Sale Price in 1000 USD

# %% [code]
plt.figure(dpi=100)
sns.distplot(housing_data_cleaned[['SalePrice']]/1000)
plt.show()

# %% [markdown]
# #### 2. Lot Area in 1000 USD

# %% [code]
plt.figure(dpi=120)
sns.distplot(housing_data_cleaned[['LotArea']]/1000)
plt.show()

# %% [markdown]
# #### 3. Year built

# %% [code]
plt.figure(dpi=120)
sns.distplot(housing_data_cleaned[['YearBuilt']])
plt.show()

# %% [markdown]
# ## Exploring Relationships between Price and Features

# %% [markdown]
# #### 1. Relationship between Price and Lot area

# %% [code]
sns.relplot(x='LotArea',y='SalePrice',hue="YearBuilt",data=housing_data_cleaned)
sns.despine()

# %% [code]
housing_data_cleaned[['LotArea','SalePrice']].corr()

# %% [markdown]
# With a correlation of **0.263843** we can say that there is a weak positive correlation between Sales Price and Lot Area

# %% [markdown]
# #### 2. Relationship between Price and Year built

# %% [code]
sns.relplot(x='YearBuilt',y='SalePrice',hue='YearBuilt',data=housing_data_cleaned)
plt.show()

# %% [code]
housing_data_cleaned[['YearBuilt','SalePrice']].corr()

# %% [markdown]
# With a corralation value of **0.522897** it indicates that there is a strong correlation between Year Built and Price

# %% [markdown]
# ## Predicting Prices base on features

# %% [markdown]
# #### 1. Import Sklearn Packages

# %% [code]
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# %% [markdown]
# #### 2. Split data into test and training

# %% [code]
### Selecting Features
train_data = housing_data_cleaned
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
target = ['SalePrice']

##split into dependent and independent variables
X = train_data[feature_names]
y = train_data[target]
##split into test and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# %% [markdown]
# #### 3. Finding the best leaf node for training

# %% [code]
from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_X, val_X, train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X,train_y)
    pred_val = model.predict(val_X)
    return mean_absolute_error(val_y,pred_val)
    

mae = 10**100
for max_leaf_nodes in [5,50,500,500]:
    current_mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Leaf Node",max_leaf_nodes,"MAE:",current_mae)
    if(mae>current_mae):
        mae = current_mae
        best_leaf_node = max_leaf_nodes

print(best_leaf_node)

# %% [markdown]
# #### 4. Train the model

# %% [code]
regressor = DecisionTreeRegressor(max_leaf_nodes=best_leaf_node,random_state=1)
regressor.fit(X,y)

# %% [markdown]
# #### 4.Predict prices from test.csv

# %% [code]
test_file_path ='/kaggle/input/iowa-house-prices/test.csv'
test_data = pd.read_csv(test_file_path)
test_data.columns

# %% [code]
test_data = test_data[feature_names]
regressor.predict(test_data)

# %% [code]
