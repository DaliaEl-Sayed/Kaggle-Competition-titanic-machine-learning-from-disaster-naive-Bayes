import numpy as np
import csv
import pandas
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn import  metrics
from  sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer



filename1 = 'train.csv'
filename2 = 'test.csv'
data_training = pandas.read_csv(filename1)
data_testing = pandas.read_csv(filename2)
df1 = pandas.DataFrame(data_training)
df2 = pandas.DataFrame(data_testing)
data = pandas.concat([df1, df2], ignore_index=True)  ##combine both test and train data set to fill the missing data in the set with the appropriate values
cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
data = data[cols] #after combining data rearrange columns order
##print(data.describe())
##print(data)
#print(data.isnull().sum()) ##get all nan numbers for each col
################# handle the FARE NAN values ################################
Fare_Nan_rows = data[data['Fare'].isnull()]
#print(Fare_Nan_rows) # find nan row
data_Fare_of_embarked_equal_s_and_pclass_equal_3 = data[(data['Pclass']==3) & (data['Embarked']=='S')].filter(items=["Fare"]) #find fare col of embarked = s and pclass = 3
median_Fare = data_Fare_of_embarked_equal_s_and_pclass_equal_3.median() #get the madian value of col fare and replace nan value with it
data['Fare'][1043] = median_Fare
################# handle Embarked missing data #########################
Embarked_Nan_rows = data[data['Embarked'].isnull()]
#print(Embarked_Nan_rows) #tl3o first class and same cabine and ticket = probability anhom ykono m3 b3d we from same port is bigger
data_emberked_of_Pclass_equal_one = data[(data['Pclass']==1)].filter(items=["Embarked"])
#print(data_emberked_of_Pclass_equal_one.groupby("Embarked").size()) #number of pclass = 1 and embarked = S is higher than C and Q so i will fill nan values with S
char_s = 'S'
data['Embarked'][829] = char_s
data.loc[data["Embarked"].isnull(), 'Embarked'] = char_s
#print(data.isnull().sum()) ##get all nan numbers for each col
################ handle age missing data ##############################
mean_Age = data["Age"].mean()
#print(mean_Age)
data.Age.fillna(mean_Age, inplace=True)
#print(data[data['Age'].isnull()])
#print(data.isnull().sum()) ##get all nan numbers for each col
################## add a new tuple for the title ########################
Title = ['Mr'] * 1309
data["Title"] = Title
data.loc[data['Name'].str.contains("Mrs."), 'Title'] = 'Mrs'
data.loc[data['Name'].str.contains("Miss. "), 'Title'] = 'Miss'
data.loc[data['Name'].str.contains("Capt. "), 'Title'] = 'Capt'
data.loc[data['Name'].str.contains("Col. "), 'Title'] = 'Col'
data.loc[data['Name'].str.contains("Don. "), 'Title'] = 'Don'
data.loc[data['Name'].str.contains("Dona. "), 'Title'] = 'Dona'
data.loc[data['Name'].str.contains("Dr. "), 'Title'] = 'Dr'
data.loc[data['Name'].str.contains("Jonkheer. "), 'Title'] = 'Jonkheer'
data.loc[data['Name'].str.contains("Lady. "), 'Title'] = 'Lady'
data.loc[data['Name'].str.contains("Major. "), 'Title'] = 'Major'
data.loc[data['Name'].str.contains("Master. "), 'Title'] = 'Master'
data.loc[data['Name'].str.contains("Mlle. "), 'Title'] = 'Mlle'
data.loc[data['Name'].str.contains("Mme. "), 'Title'] = 'Mme'
data.loc[data['Name'].str.contains("Ms. "), 'Title'] = 'Ms'
data.loc[data['Name'].str.contains("Rev. "), 'Title'] = 'Rev'
data.loc[data['Name'].str.contains("Sir. "), 'Title'] = 'Sir'
data.loc[data['Name'].str.contains("The Countess. "), 'Title'] = 'The Countess'
###################### reduce titles to master miss mr mrs other ####################################
data.loc[data['Title'].str.contains("Mme"), 'Title'] = 'Mrs'
data.loc[data['Title'].str.contains("Ms")| data['Title'].str.contains("Mlle"), 'Title'] = 'Miss'
data.loc[data['Title'].str.contains("Dona")| data['Title'].str.contains("Dr")| data['Title'].str.contains("Lady")|
         data['Title'].str.contains("Capt")| data['Title'].str.contains("Col")| data['Title'].str.contains("Don")|
         data['Title'].str.contains("Major")| data['Title'].str.contains("Rev")| data['Title'].str.contains("Sir")|
         data['Title'].str.contains("The Countess")| data['Title'].str.contains("Jonkheer"), 'Title'] = 'Other'
#print(data.groupby("Title").size())
###################### add new feature of family size##################################################################
FamilySize = ["Single"] * 1309
data["FamilySize"] = FamilySize
#print(data.groupby("FamilySize").size())
FamilySize1 = [1] * 1309
data["FamilySize1"] = FamilySize1
data['FamilySize1'] = data['Parch'] + data['SibSp'] + 1
#print(data.groupby("FamilySize1").size()) #too many i will reduce it to 3 groups single - large - single - small
data.loc[(data['FamilySize1'] == 2 ) | (data['FamilySize1'] == 3) | (data['FamilySize1'] == 4), 'FamilySize'] = 'Small'
data.loc[(data['FamilySize1'] == 5 ) | (data['FamilySize1'] == 6) | (data['FamilySize1'] == 7) | (data['FamilySize1'] == 8)
         | (data['FamilySize1'] == 11), 'FamilySize'] = 'Large'
data = data.drop('FamilySize1', 1)
#print(data.groupby("FamilySize").size())
##################### FUN Time :D #########################################################################
################### convert text to num data ##############################################
vect = CountVectorizer()
vect.fit(data["FamilySize"])
vect.get_feature_names()
train_data_familysize_convert = vect.transform(data["FamilySize"])
new_Family_size = pandas.DataFrame(train_data_familysize_convert.toarray(), columns=vect.get_feature_names())
data = data.drop('FamilySize', 1)
data = pandas.concat([data, new_Family_size],axis=1)

vect2 = CountVectorizer()
vect2.fit(data["Sex"])
vect2.get_feature_names()
train_data_sex_convert = vect2.transform(data["Sex"])
new_sex = pandas.DataFrame(train_data_sex_convert.toarray(), columns=vect2.get_feature_names())
data = data.drop('Sex', 1)
data = pandas.concat([data, new_sex],axis=1)
data = data.drop('Cabin', 1)
data = data.drop('Embarked', 1)

vect4 = CountVectorizer()
vect4.fit(data["Title"])
vect4.get_feature_names()
train_data_Title_convert = vect4.transform(data["Title"])
new_Title = pandas.DataFrame(train_data_Title_convert.toarray(), columns=vect4.get_feature_names())
data = data.drop('Title', 1)
data = pandas.concat([data, new_Title], axis=1)

vect5 = CountVectorizer()
vect5.fit(data["Ticket"])
vect5.get_feature_names()
train_data_Ticket_convert = vect5.transform(data["Ticket"])
new_Ticket = pandas.DataFrame(train_data_Ticket_convert.toarray(), columns=vect5.get_feature_names())
data = data.drop('Ticket', 1)
data = pandas.concat([data, new_Ticket],axis=1)

vect6 = CountVectorizer()
vect6.fit(data["Name"])
vect6.get_feature_names()
train_data_Name_convert = vect6.transform(data["Name"])
new_Name = pandas.DataFrame(train_data_Name_convert.toarray(), columns=vect6.get_feature_names())
data = data.drop('Name', 1)
data = pandas.concat([data, new_Name],axis=1)

data.loc[data["Survived"].isnull(), 'Survived'] = 0
########################### data = 2 parts  again #################################################
training_data = data[:891]
testing_data = data[891:]
survived = training_data["Survived"]
final_prediction = testing_data[["PassengerId", "Survived"]].copy()
########################## time to predict ################################
GNB = GaussianNB()
GNB.fit(training_data, survived)
survived_predict = GNB.predict(testing_data).astype(int)
final_prediction["Survived"] = survived_predict
print(final_prediction[["PassengerId", "Survived"]])
final_prediction.reset_index(inplace=True)
header = ["PassengerId", "Survived"]
final_prediction.to_csv("final_prediction_result.csv", sep=",", columns=header, index=False)

###################################### The End XD ##############################################