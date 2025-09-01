# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:

## Program to implement the linear regression using gradient descent.
Developed by: Priyadharshini E 
RegisterNumber:  2122223230159

```

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
        pass
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

X=(data.iloc[1:, :-2].values)
print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")



```






## Output:


<img width="720" height="311" alt="image" src="https://github.com/user-attachments/assets/e03bb6ae-46a6-42ab-bc9e-2a0f506bc4a0" />

<img width="1094" height="303" alt="image" src="https://github.com/user-attachments/assets/38f6f028-0f30-4b57-bd54-3f43cc4ae838" />

<img width="766" height="299" alt="image" src="https://github.com/user-attachments/assets/649aba9a-4d77-42c2-89ad-a7d8e9d9e8c1" />

<img width="685" height="314" alt="image" src="https://github.com/user-attachments/assets/a8147ecc-023f-4f15-b0cf-b4c989dcd7a6" />

<img width="411" height="304" alt="image" src="https://github.com/user-attachments/assets/7ec48c69-6fd6-4335-8b92-40708db603dd" />

<img width="335" height="303" alt="image" src="https://github.com/user-attachments/assets/0cc489fb-b172-4b44-a99d-d366af0bddc3" />

<img width="559" height="313" alt="image" src="https://github.com/user-attachments/assets/5f934adf-84cc-4305-86a7-389f351a8762" />

<img width="613" height="302" alt="image" src="https://github.com/user-attachments/assets/d8795fe4-8fb1-41aa-a84b-6e6b6a2426d5" />

<img width="484" height="312" alt="image" src="https://github.com/user-attachments/assets/1cac16bf-b6ed-4cfd-b085-6286716c8aa2" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
