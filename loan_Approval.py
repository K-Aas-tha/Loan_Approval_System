'''importing all necessary libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

'''fetching data '''

df=pd.read_csv("loan_approval_data.csv")
# print(df.head())
# print(df.info())
# print(df.isnull().sum())

'''Data Preprocessing '''
category=df.select_dtypes(include=["str"]).columns
numerical=df.select_dtypes(include=["float64"]).columns

num_imp=SimpleImputer(strategy="mean")
df[numerical]=num_imp.fit_transform(df[numerical])
# print(df.head())
cat_imp=SimpleImputer(strategy="most_frequent")
df[category]=cat_imp.fit_transform(df[category])
# print(df.head())
# print(df.isnull().sum())

''' Performing EDA'''
sns.histplot(
    data=df,
    x="Applicant_Income",
    bins=20
)
# plt.show()

fig,axes=plt.subplots(2,3)
sns.boxplot(ax=axes[0,0],data=df,x="Loan_Approved",y="Applicant_Income")
sns.boxplot(ax=axes[0,1],data=df,x="Loan_Approved",y="Credit_Score")
sns.boxplot(ax=axes[0,2],data=df,x="Loan_Approved",y="DTI_Ratio")
sns.boxplot(ax=axes[1,0],data=df,x="Loan_Approved",y="Savings")
sns.boxplot(ax=axes[1,1],data=df,x="Loan_Approved",y="Age")
sns.boxplot(ax=axes[1,2],data=df,x="Loan_Approved",y="Loan_Amount")
plt.tight_layout()
plt.show()


df=df.drop("Applicant_ID",axis=1)
# # print(df.head())
# print(df.columns)

'''FEATURE ENCODING'''
le=LabelEncoder()
df["Education_Level"]=le.fit_transform(df["Education_Level"])
df["Loan_Approved"]=le.fit_transform(df["Loan_Approved"])
# print(df.head())

col_s=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Employer_Category","Gender"]
ohe=OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore")
encoded=ohe.fit_transform(df[col_s])
df_e=pd.DataFrame(encoded,columns=ohe.get_feature_names_out(col_s),index=df.index)
# print(df_e.head())
# print(df_e.columns)

df=pd.concat([df.drop(columns=col_s),df_e],axis=1)
# print(df.head())
# print(df.info())

'''CORRELATION HEATMAP(to define co -relations)'''
num_cols=df.select_dtypes(include="number")
corr_mtx=num_cols.corr()
plt.figure()
sns.heatmap(
    corr_mtx,
    annot=True,
    fmt="0.2f",
    cmap="coolwarm"
)
plt.show()

'''TRAINING our MODEL'''
X=df.drop("Loan_Approved",axis=1)
y=df["Loan_Approved"]
# print(y.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)

# print(X_train_scaled)

'''implementing LOGISTIC Regression'''
lo_model=LogisticRegression(max_iter=100)
lo_model.fit(X_train_scaled,y_train)
y_pred=lo_model.predict(X_test_scaled)
print(f"Accuracy : {accuracy_score(y_test,y_pred)*100}%")
print(f"Precision : {precision_score(y_test,y_pred)*100}%")
print(f"Recall : {recall_score(y_test,y_pred)*100}%")
print(f"f1_score : {f1_score(y_test,y_pred)*100}%")
print(f"CM : {confusion_matrix(y_test,y_pred)}")

'''PERFORMING naive bayes '''
nb= GaussianNB()
nb.fit(X_train_scaled,y_train)
y_pred=nb.predict(X_test_scaled)
print(" # Naive_Bayes")
print(f"Accuracy : {accuracy_score(y_test,y_pred)*100}%")
print(f"Precision : {precision_score(y_test,y_pred)*100}%")
print(f"Recall : {recall_score(y_test,y_pred)*100}%")
print(f"f1_score : {f1_score(y_test,y_pred)*100}%")
print(f"CM : {confusion_matrix(y_test,y_pred)}")

'''Testing for KNN'''
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled,y_train)
y_pred=knn.predict(X_test_scaled)
print("# knn")
print(f"Accuracy : {accuracy_score(y_test,y_pred)*100}%")
print(f"Precision : {precision_score(y_test,y_pred)*100}%")
print(f"Recall : {recall_score(y_test,y_pred)*100}%")
print(f"f1_score : {f1_score(y_test,y_pred)*100}%")
print(f"CM : {confusion_matrix(y_test,y_pred)}")

# FEATURE ENGINEERING 
'''improving features that have correlations(dti ratio and credit score)'''
print(df.columns)
df["DTI_Ratio_squared"]=df["DTI_Ratio"]**2
df["Credit_Score_squared"]=df["Credit_Score"]**2
X=df.drop(columns=["Credit_Score","DTI_Ratio","Loan_Approved"])
y=df["Loan_Approved"]

df["Applicant_Income_log"]=np.log1p(df["Applicant_Income"])
X=df.drop(columns=["Credit_Score","DTI_Ratio","Loan_Approved","Applicant_Income"])
y=df["Loan_Approved"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)

'''PERFORMING LOGISTIC REGRESSION,KNN,AND NAIVE_BAYES'''
# LOGISTIC REGRESSION
lo_model=LogisticRegression(max_iter=100)
lo_model.fit(X_train_scaled,y_train)
y_pred=lo_model.predict(X_test_scaled)
print(f"Accuracy : {accuracy_score(y_test,y_pred)*100}%")
print(f"Precision : {precision_score(y_test,y_pred)*100}%")
print(f"Recall : {recall_score(y_test,y_pred)*100}%")
print(f"f1_score : {f1_score(y_test,y_pred)*100}%")
print(f"CM : {confusion_matrix(y_test,y_pred)}")

# NAIVE BAYES 
nb= GaussianNB()
nb.fit(X_train_scaled,y_train)
y_pred=nb.predict(X_test_scaled)
print(" # Naive_Bayes")
print(f"Accuracy : {accuracy_score(y_test,y_pred)*100}%")
print(f"Precision : {precision_score(y_test,y_pred)*100}%")
print(f"Recall : {recall_score(y_test,y_pred)*100}%")
print(f"f1_score : {f1_score(y_test,y_pred)*100}%")
print(f"CM : {confusion_matrix(y_test,y_pred)}")

# KNN
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled,y_train)
y_pred=knn.predict(X_test_scaled)
print("# knn")
print(f"Accuracy : {accuracy_score(y_test,y_pred)*100}%")
print(f"Precision : {precision_score(y_test,y_pred)*100}%")
print(f"Recall : {recall_score(y_test,y_pred)*100}%")
print(f"f1_score : {f1_score(y_test,y_pred)*100}%")
print(f"CM : {confusion_matrix(y_test,y_pred)}")

#overall better model for this is naive bayes based on precision score 
