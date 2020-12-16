# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:22:04 2020

@author: aaron
"""
import pandas as pd
import numpy as np
data = pd.read_csv("diamonds.csv")

def Task1():
    print(data.head())
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
                Price = amount[["price"]].values
                if len(amount) >=800:
                    Feature_Dict.update({Key:value})
                    Target_Dict.update({Key:Price})
                    
    print(Feature_Dict)
    print(Target_Dict)
    
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
    
def eval_poly_3(d,a,x,y,z):
    r = 0
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        r += a[t]*(x**i)*(y**j)*(z**k)
                        t = t+1
    return r

def calculate_model_function(deg,data, p):
    result = np.zeros(data.shape[0])    
    k=0
    for n in range(deg+1):
        for i in range(n+1):
            result += p[k]*(data[:,0]**i)*(data[:,1]**(n-i))
            k+=1
    return result

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
    Degree = 3
    p0 = regression(Degree,Features,Targets)
    
        
        