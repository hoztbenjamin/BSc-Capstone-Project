# This was the code used for the data preparation and data sampling process of my FYP
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV, Tab seperator, thousands seperator present
df = pd.read_csv('cleaneddata.txt', sep='\t', thousands=",")

cols = [0, 1, 2]
df1 = df.drop(df.columns[cols], axis= 1)
#drop name, year, and index

replace = df1["Bankruptcy Status"].map({"BANKRUPT":1, "Non-bankrupt":0})
df1["Bankruptcy Status"] = replace
#recode target variable with 1 and 0

df1 = df1.dropna(thresh= int(0.7*len(df)), axis=1)
#drop features which have less than 70% completeness 

df2 = df1.fillna(df.mean())
#fill missing values with mean value

features = df2.columns.tolist()[:-1]
#set features as all but "bankruptcy status"

target = df2.columns.tolist()[-1]
#set target as "bankruptcy status"
y = df2[[target]]
x = df2[features]

x.dtypes
#check data types
df2
df2.isnull().sum() * 100 / len(df)
# Plot distribution of Bankruptcy Status

ax = df2[target].value_counts().plot(kind='bar', figsize=(10, 6), fontsize=13, color='#087E8B')
ax.set_title('DIstribution of Bankruptcy Status', size=20, pad=30)
ax.set_ylabel('Number of Records', fontsize=14)
ax.set_ylim(0,5500)

for i in ax.patches:
    ax.text(i.get_x() + 0.19, i.get_height() + 200, str(round(i.get_height(), 2)), fontsize=15)
# Imports
import numpy as np
# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

# Split train/test dataset to 80:20
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)

# ADASYN sampling technique (on training data set only)
ads = ADASYN(sampling_strategy=0.25,random_state=44)
x_ads, y_ads = ads.fit_resample(x_train, y_train)

# # Smote sampling technique (on training dataset only)
# sm = SMOTE(sampling_strategy=0.25,random_state=44)
# x_sm, y_sm = sm.fit_resample(x, y)


# Convert smote output into dataframe (change for ADASYN or SMOTE accordingly)
df_train = pd.DataFrame(x_ads, columns=x.columns)
df_train['Bankruptcy Status'] = y_ads

df_test = pd.DataFrame(x_test,columns=x.columns)
df_test['Bankruptcy Status'] = y_test

# Fill blanks with column's mean
df_test.fillna(df_test.mean(), inplace=True)

# Check null values
df_train.isnull().sum()

# Check data types
df.dtypes

# Exports out in csv file as tsv format
df_train.to_csv('adasyn_train.csv', encoding='utf-8', sep='\t')
df_test.to_csv('adasyn_test.csv', encoding='utf-8', sep='\t')


print(f'''Shape of X before SMOTE: {x.shape}
Shape of X after SMOTE: {x_ads.shape}''')

print("\033[1m"+"Balance of positive and negative classes (%):" + "\033[0m")
y_ads["Bankruptcy Status"].value_counts(normalize=True) * 100
# Graphical representation of class propotion (just change the ax to whatever column u want to represent)
ax = y_ads["Bankruptcy Status"].value_counts().plot(kind='bar', figsize=(10, 6), fontsize=13, color='#087E8B')
ax.set_title('Distribution of Bankruptcy Status', size=20, pad=30)
ax.set_ylabel('Number of Records', fontsize=14)
ax.set_ylim(0,5500)

for i in ax.patches:
    ax.text(i.get_x() + 0.19, i.get_height() + 200, str(round(i.get_height(), 2)), fontsize=15)

print("\033[1m"+"Balance of positive and negative classes (%):" + "\033[0m")
y_ads["Bankruptcy Status"].value_counts(normalize=True) * 100
