# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:22:04 2020

@author: aaron
"""
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv("diamonds.csv")

def Task1():
    # print(data.head())
    Quality = data[["cut","color","clarity"]]
    Cut = Quality[["cut"]]
    Color = Quality[["color"]]
    Clarity =Quality[["clarity"]]
    
    Cut_Unique = Quality["cut"].unique()
    Color_Unique = Quality["color"].unique()
    Clarity_Unique = Quality["clarity"].unique()
    # print(Cut_Unique)
    # print(Color_Unique)
    # print(Clarity_Unique)
    # print(Cut.head())
    # print(Color.head())
    # print(Clarity.head())
    
    Target_Dict = {}
    Feature_Dict = {}
    
    for Cut in Cut_Unique:
        for Color in Color_Unique:
            for Clarity in Clarity_Unique:
                Key = Cut + "," + Color + "," + Clarity
                amount = data[data["cut"]==Cut][data["color"]==Color][data["clarity"]==Clarity]
                value = amount[["carat","table","depth"]].values
                Price = amount["price"].values
                if len(amount) >=800:
                    Feature_Dict.update({Key:value})
                    Target_Dict.update({Key:Price})
                    
    # print(Feature_Dict)
    # print(Target_Dict)
    
    return Feature_Dict, Target_Dict
                


def linearize(deg,data, p0):
    f0 = calculate_model_function(deg,data,p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg,data,p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
    return f0,J


def num_coefficients_3(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        t = t+1
    return t


def calculate_model_function(deg,data, p):
    r = 0
    t = 0
    for n in range(deg+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        r += p[t]*(data[:,0]**i)*(data[:,1]**j)*(data[:,2]**k)
                        t = t+1
    return r

def calculate_update(y,f0,J):
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)    
    dp = np.linalg.solve(N,n)       
    return dp


def regression(deg,data,target):
    max_iter = 10
   
    p0 = np.zeros(num_coefficients_3(deg))
    for i in range(max_iter):
        f0,J = linearize(deg,data, p0)
        dp = calculate_update(target,f0,J)
        p0 += dp
    return p0


def main():
    Features, Targets = Task1()
    kf = model_selection.KFold(n_splits=4, shuffle=True)
    List_best = [0,0,0,0,0]
    count = 0
    Degree_Zero = []
    Degree_One = []
    Degree_Two = []
    Degree_Three =[]
    
    Degree_Zero_p0 = []
    Degree_One_p0 = []
    Degree_Two_p0 = []
    Degree_Three_p0 =[]
    for f,t in zip(Features, Targets): 
        for deg in range(0,4):
            for train_index, test_index in kf.split(Features[f]): 
                p0 = regression(deg,Features[f][train_index],Targets[t][train_index])
        
                predict = calculate_model_function(deg, Features[f][test_index],p0)
                Actual_Predicted = abs(np.mean(Targets[t]) - np.mean(predict))
                if(deg==0):
                    Degree_Zero.append(Actual_Predicted)
                    Degree_Zero_p0.append(p0)
                elif(deg==1):
                    Degree_One.append(Actual_Predicted)
                    Degree_One_p0.append(p0)
                elif(deg==2):
                    Degree_Two.append(Actual_Predicted)
                    Degree_Two_p0.append(p0)
                else:
                    Degree_Three.append(Actual_Predicted)
                    Degree_Three_p0.append(p0)


                # print(Actual_Predicted)
            
        min_val = 1000
        for i in Degree_Zero:
            if i > min_val:
                min_val = i
                List_best[count] = 0
        
        for j in Degree_One:
            if j < min_val:
                min_val = j
                List_best[count] = 1
                
        for k in Degree_Two:
            if k < min_val:
                min_val = k
                List_best[count] = 2
                
        for l in Degree_Three:
            if l < min_val:
                min_val = l
                List_best[count] = 3
        
        count = count+1

    print(List_best)
main()

# def plot():
#     x = Targets[t][test_index]
#     y = predict

#     plt.figure()
    
#     plt.scatter(x, y)
#     plt.xlabel("True Prices")
#     plt.ylabel("Predicted Prices")
#     plt.show()