<H3>NAME: SETHUKKARASI C</H3>
<H3>REGISTER NUMBER: 212223230201</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 25.08.2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM & OUTPUT:

```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```

```
#Read the dataset from drive
df = pd.read_csv('Churn_Modelling.csv')
print(df)
```

<img width="692" height="809" alt="image" src="https://github.com/user-attachments/assets/5583a050-89c1-47a3-a793-c5caa9f50bcd" />

```
#Split the dataset
X = df.iloc[:,:-1].values
print(X)
Y = df.iloc[:,-1].values
print(Y)
```

<img width="459" height="274" alt="image" src="https://github.com/user-attachments/assets/d61eebb9-86ae-48a1-9ece-dccc7e61a3e2" />

```
# Finding Missing Values
print(df.isnull().sum())
```

<img width="268" height="346" alt="image" src="https://github.com/user-attachments/assets/9348c1da-0067-49ec-805a-222a6b33de60" />

```
#Dropping string values data from dataset
df = df.drop(['Surname', 'Geography','Gender'], axis=1)
df.head()
```

<img width="1135" height="292" alt="image" src="https://github.com/user-attachments/assets/1088c117-47c1-4de8-a0ec-6104358888ae" />

```
#Handling Missing values
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
Y = df.iloc[:,-1].values
print(Y)
```

<img width="427" height="360" alt="image" src="https://github.com/user-attachments/assets/6f9467af-db03-4d0d-ad0e-39ce1f6fcd11" />

```
#Check for Duplicates
df.duplicated()
```

<img width="310" height="532" alt="image" src="https://github.com/user-attachments/assets/452d6bbf-58b6-4397-b1d5-7969a8d3d367" />

```
#Detect Outliers
print(df.describe())
```

<img width="671" height="574" alt="image" src="https://github.com/user-attachments/assets/2011e748-077d-4207-9c52-544d56dbd758" />

```
#Normalize the dataset
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
```

<img width="675" height="590" alt="image" src="https://github.com/user-attachments/assets/8df6624e-49a3-4010-b28b-b72a97135c03" />

```
#split the dataset into input and output
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```

<img width="628" height="386" alt="image" src="https://github.com/user-attachments/assets/b59ccea7-11cf-49d2-a13a-0a100b15c778" />

```
#splitting the data for training & Testing
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
```

```
#Print the training data and testing data
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

<img width="602" height="784" alt="image" src="https://github.com/user-attachments/assets/98fa9070-ae7c-49b3-969d-96388d500706" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


