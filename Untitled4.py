#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
df =pd.read_csv(r"C:\Users\Venkkaiah Chowdary\Downloads\data.csv")
df


# In[3]:


df.mean()


# In[4]:


df.median()


# In[5]:


df.mode()


# In[6]:


df.std()


# In[7]:


df.var()


# In[8]:


df.isnull().sum()


# In[11]:


df['Calories'] = df['Calories'].fillna(df['Calories'].mean())


# In[12]:


df.isnull().sum()


# In[13]:


result = df[['Duration','Maxpulse']].agg(['min','max','count','mean'])
result


# In[14]:


d1 = df[df['Calories'].between(500,1000)]
d1


# In[16]:


d2 = df[(df['Calories']>500) & (df['Pulse']<100)]
d2


# In[17]:


df_modified=df.drop('Maxpulse',axis=1)
df_modified


# In[18]:


df.drop('Maxpulse',axis=1)


# In[19]:


df["Calories"]=df["Calories"].astype(float).astype(int)
df


# In[20]:


plot = df.plot.scatter(x="Duration",y="Calories")


# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
salary = pd.read_csv(r"C:\Users\Venkkaiah Chowdary\Downloads\Salary_Data.csv")
salary


# In[22]:


X = salary["YearsExperience"]
Y = salary["Salary"]
#X1 = [[i,x] for i, x in enumerate(X)]
#Y1 = [[i,y] for i, y in enumerate(Y)]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)


# In[23]:


regressor = LinearRegression() 
model = regressor.fit(X_train.values.reshape(-1, 1),Y_train.values.reshape(-1, 1))


# In[24]:


print(model.coef_)
print(model.intercept_)


# In[25]:


Y_predict = model.predict(X_test.values.reshape(-1,1))
plt.title("Salary/Years of XP")
plt.ylabel("Salary $")
plt.xlabel("Years")
plt.scatter(X_test,Y_test,color="blue",label="real data")
plt.plot(X_test,Y_predict,color="red",label="linear model")
plt.legend()
plt.show()


# In[26]:


mean_squared_error(Y_test, Y_predict)


# In[ ]:




