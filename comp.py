import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import category_encoders as ce
from scipy import stats

dataset = pd.read_csv('/Users/svetl/OneDrive/Desktop/4th Year/ML/tcd ml 2019-20 income prediction training (with labels).csv')
tryset = pd.read_csv('/Users/svetl/OneDrive/Desktop/4th Year/ML/tcd ml 2019-20 income prediction test (without labels).csv')
instanceSet = pd.read_csv('/Users/svetl/OneDrive/Desktop/4th Year/ML/tcd ml 2019-20 income prediction submission file.csv')

dataset.loc[dataset['Gender'] == 'male', 'Gender'] = 1
dataset.loc[dataset['Gender'] == 'female', 'Gender'] = 2
dataset.loc[dataset['Gender'] == 'other', 'Gender'] = 3
dataset.loc[dataset['Gender'] == 'unknown', 'Gender'] = 4

dataset.loc[dataset['University Degree'] == 'No', 'University Degree'] = 1
dataset.loc[dataset['University Degree'] == 'Bachelor', 'University Degree'] = 2
dataset.loc[dataset['University Degree'] == 'Master', 'University Degree'] = 3
dataset.loc[dataset['University Degree'] == 'PhD', 'University Degree'] = 4

dataset.loc[dataset['Hair Color'] == 'Blond', 'Hair Color'] = 1
dataset.loc[dataset['Hair Color'] == 'Black', 'Hair Color'] = 2
dataset.loc[dataset['Hair Color'] == 'Brown', 'Hair Color'] = 3
dataset.loc[dataset['Hair Color'] == 'Red', 'Hair Color'] = 4
dataset.loc[dataset['Hair Color'] == 'Unknown', 'Hair Color'] = 5

tryset.loc[tryset['Gender'] == 'male', 'Gender'] = 1
tryset.loc[tryset['Gender'] == 'female', 'Gender'] = 2
tryset.loc[tryset['Gender'] == 'other', 'Gender'] = 3
tryset.loc[tryset['Gender'] == 'unknown', 'Gender'] = 4

tryset.loc[tryset['University Degree'] == 'No', 'University Degree'] = 1
tryset.loc[tryset['University Degree'] == 'Bachelor', 'University Degree'] = 2
tryset.loc[tryset['University Degree'] == 'Master', 'University Degree'] = 3
tryset.loc[tryset['University Degree'] == 'PhD', 'University Degree'] = 4

tryset.loc[tryset['Hair Color'] == 'Blond', 'Hair Color'] = 1
tryset.loc[tryset['Hair Color'] == 'Black', 'Hair Color'] = 2
tryset.loc[tryset['Hair Color'] == 'Brown', 'Hair Color'] = 3
tryset.loc[tryset['Hair Color'] == 'Red', 'Hair Color'] = 4
tryset.loc[tryset['Hair Color'] == 'Unknown', 'Hair Color'] = 5

dataset['Gender'] = dataset['Gender'].fillna(0)
dataset['University Degree'] = dataset['University Degree'].fillna(0)
dataset['Hair Color'] = dataset['Hair Color'].fillna(0)

tryset['Gender'] = tryset['Gender'].fillna(0)
tryset['University Degree'] = tryset['University Degree'].fillna(0)
tryset['Hair Color'] = tryset['Hair Color'].fillna(0)

dataset['Age'] = dataset['Age'].fillna(dataset.Age.mean())
dataset['Year of Record'] = dataset['Year of Record'].fillna(dataset['Year of Record'].mean())
tryset['Age'] = tryset['Age'].fillna(tryset.Age.mean())
tryset['Year of Record'] = tryset['Year of Record'].fillna(tryset['Year of Record'].mean())

dataset = dataset.fillna(method='ffill')
tryset = tryset.fillna(method='ffill')

profListData = dataset.Profession.unique()
profListTry = tryset.Profession.unique()
combined = np.concatenate((profListTry, profListData), axis=0)
profList = np.unique(combined)
length = len(profList)
pInd = 0

for prof in profListData:
    tryset.loc[tryset['Profession'] == prof, 'Profession'] = pInd
    dataset.loc[dataset['Profession'] == prof, 'Profession'] = pInd
    pInd = pInd+1

couListData = dataset.Country.unique()
couListTry = tryset.Country.unique()
combined = np.concatenate((couListTry, couListData), axis=0)
couList = np.unique(combined)
length = len(couList)
ind = 0

for cou in couListData:
    tryset.loc[tryset['Country'] == cou, 'Country'] = ind
    dataset.loc[dataset['Country'] == cou, 'Country'] = ind
    ind = ind+1

tryset.loc[tryset['Country'] == 'Iceland', 'Country'] = ind
tryset.loc[tryset['Country'] == 'Italy', 'Country'] = ind+1
tryset.loc[tryset['Country'] == 'Turkey', 'Country'] = ind+2
tryset.loc[tryset['Country'] == 'Samoa', 'Country'] = ind+3
tryset.loc[tryset['Country'] == 'Sao Tome & Principe', 'Country'] = ind+4
tryset.loc[tryset['Country'] == 'France', 'Country'] = ind+5
tryset.loc[tryset['Profession'] == 'asset management specialist', 'Profession'] = pInd
tryset.loc[tryset['Profession'] == 'administrative manager', 'Profession'] = pInd+1
tryset.loc[tryset['Profession'] == 'astronomer', 'Profession'] = pInd+2
tryset.loc[tryset['Profession'] == 'brokerage clerk', 'Profession'] = pInd+3
tryset.loc[tryset['Profession'] == 'asset manager', 'Profession'] = pInd+4
tryset.loc[tryset['Profession'] == 'certified it administrator', 'Profession'] = pInd+5
tryset.loc[tryset['Profession'] == 'cashier', 'Profession'] = pInd+6
tryset.loc[tryset['Profession'] == 'aerospace engineer', 'Profession'] = pInd+7
tryset.loc[tryset['Profession'] == 'apparel patternmaker', 'Profession'] = pInd+8
tryset.loc[tryset['Profession'] == 'clinical case supervisor', 'Profession'] = pInd+9
tryset.loc[tryset['Profession'] == 'baggage porter', 'Profession'] = pInd+10

dataset = pd.get_dummies(dataset, columns=['Wears Glasses'], prefix = ['Wears Glasses'])
tryset = pd.get_dummies(tryset, columns=['Wears Glasses'], prefix = ['Wears Glasses'])
dataset = pd.get_dummies(dataset, columns=['Hair Color'], prefix = ['Hair Color'])
tryset = pd.get_dummies(tryset, columns=['Hair Color'], prefix = ['Hair Color'])

dataset['Interaction1'] = dataset['Profession'].astype(int)*dataset['Gender'].astype(int)
tryset['Interaction1'] = tryset['Profession'].astype(int)*tryset['Gender'].astype(int)
dataset['Interaction2'] = dataset['Profession'].astype(int)*dataset['University Degree'].astype(int)
tryset['Interaction2'] = tryset['Profession'].astype(int)*tryset['University Degree'].astype(int)
dataset['Interaction3'] = dataset['Gender'].astype(int)*dataset['University Degree'].astype(int)
tryset['Interaction3'] = tryset['Gender'].astype(int)*tryset['University Degree'].astype(int)
dataset['Interaction4'] = dataset['Size of City'].astype(int)*dataset['University Degree'].astype(int)
tryset['Interaction4'] = tryset['Size of City'].astype(int)*tryset['University Degree'].astype(int)

dataset = pd.get_dummies(dataset, columns=['Gender'], prefix = ['Gender'])
tryset = pd.get_dummies(tryset, columns=['Gender'], prefix = ['Gender'])
dataset = pd.get_dummies(dataset, columns=['University Degree'], prefix = ['University Degree'])
tryset = pd.get_dummies(tryset, columns=['University Degree'], prefix = ['University Degree'])

'''
out = dataset[['Interaction1', 'Interaction2', 'Interaction3', 'Interaction4']].values
z = np.abs(stats.zscore(out))
pScore = stats.norm.cdf(z)
dataset = dataset[(pScore < 0.95).all(axis=1)]
'''
X = dataset[['Year of Record', 'Age', 'Gender_0', 'Gender_1', 'Gender_2', 'Gender_3', 'Profession', 'Country', 'Size of City', 'University Degree_0', 'University Degree_1', 'University Degree_2', 'University Degree_3', 'University Degree_4', 'Hair Color_1', 'Hair Color_2', 'Hair Color_3', 'Hair Color_4', 'Hair Color_5', 'Hair Color_0', 'Wears Glasses_0', 'Wears Glasses_1', 'Body Height [cm]', 'Interaction1', 'Interaction2', 'Interaction3', 'Interaction4']].to_numpy()
y = dataset['Income in EUR'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X, y)

yTry = tryset[['Year of Record', 'Age', 'Gender_0', 'Gender_1', 'Gender_2', 'Gender_3', 'Profession', 'Country', 'Size of City', 'University Degree_0', 'University Degree_1', 'University Degree_2', 'University Degree_3', 'University Degree_4', 'Hair Color_1', 'Hair Color_2', 'Hair Color_3', 'Hair Color_4', 'Hair Color_5', 'Hair Color_0', 'Wears Glasses_0', 'Wears Glasses_1', 'Body Height [cm]', 'Interaction1', 'Interaction2', 'Interaction3', 'Interaction4']].to_numpy()
y_predTest = regressor.predict(yTry)

yTry = tryset[['Year of Record', 'Age', 'Gender_0', 'Gender_1', 'Gender_2', 'Gender_3', 'Profession', 'Country', 'Size of City', 'University Degree_0', 'University Degree_1', 'University Degree_2', 'University Degree_3', 'University Degree_4', 'Hair Color_1', 'Hair Color_2', 'Hair Color_3', 'Hair Color_4', 'Hair Color_5', 'Hair Color_0', 'Wears Glasses_0', 'Wears Glasses_1', 'Body Height [cm]', 'Interaction1', 'Interaction2', 'Interaction3', 'Interaction4']].to_numpy()

y_pred = regressor.predict(yTry)
df = pd.DataFrame({'Income' : y_pred})
instanceSet = instanceSet.drop(instanceSet.columns[1], axis=1)
horizontal_stack = pd.concat([instanceSet, df], axis=1)
horizontal_stack.to_csv('/Users/svetl/OneDrive/Desktop/thing.csv')

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

