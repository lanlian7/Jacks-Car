#coding:utf8
'''
Created on 2018年4月21日

@author: Doris
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
import time
from _overlapped import NULL

#poission cache ,for it frequently uses
poissonBackup = dict()
def poission(n,lam):
    global poissonBackup
    key = n*10 + lam
    if key not in poissonBackup:
        poissonBackup[key] = exp(-lam)*pow(lam,n)/factorial(n);
    return poissonBackup[key]

figureIndex = 0

class JackCar:
    def __init__(self,maxCars = 20,maxMoveOfCarr = 5,rentalRequestFirstLoc = 3,rentalRequestSecondLoc = 4,returnsFirstLoc = 3,returnsSecondLoc = 2,
                 disCount = 0.9,rentalCredit = 10,moveCarCost = 2,theta = 0.001):
        self.max_cars =maxCars
        self.max_move_of_cars = maxMoveOfCarr
        self.rental_request_first_loc = rentalRequestFirstLoc
        self.rental_request_second_loc = rentalRequestSecondLoc
        self.returns_first_loc = returnsFirstLoc
        self.returns_second_loc = returnsSecondLoc
        
        self.discount = disCount
        
        self.rental_credit = rentalCredit
        self.move_car_cost = moveCarCost
        
        #逼近策略中的逼近值
        self.theta = theta
        
        self.improvePolicy = False
        self.policyImprovementInd = 0
        self.valueImprovementInd = 0
        self.diff = 0.0
        self.policyDiff = 0
        self.endImprove = False
        
        self.policy = np.zeros((self.max_cars+1,self.max_cars+1))
        
        self.stateValue = np.zeros((self.max_cars+1,self.max_cars+1))
        
        #actions :positive if moving cars from first location to second location
        #         negative if moving cars from second location to first location
        self.actions = np.arange(-self.max_move_of_cars,self.max_move_of_cars+1)
        
        #init state
        self.states = []
        for i in range(0,self.max_cars+1):
            for j in range(0,self.max_cars+1):
                self.states.append([i,j])
                
    #start policy improvement
    def improvement(self):
        print ('Policy improvement',self.policyImprovementInd)
        self.policyImprovementInd +=1
        
        newPolicy = np.zeros((self.max_cars+1,self.max_cars+1))
        for i,j in self.states:
            actionReturns = []
            for action in self.actions:
                if(action>=0 and i>=action) or(action<0 and j>=abs(action)):
                    returns = self.expectedReturn([i,j],action)
                    actionReturns.append(returns)
                else:
                    actionReturns.append(-float('inf'))
                    
            bestAction = np.argmax(actionReturns)
            newPolicy[i,j] = self.actions[bestAction]   
        # if policy is stable
        policyChanges = np.sum(newPolicy != self.policy)
        print('Policy for', policyChanges, 'states changed')
        if policyChanges == 0:
            self.policy = newPolicy
            self.endImprove = True
        self.policy = newPolicy
        self.improvePolicy = False             
        
    #start policy evaluation            
    def evaluation(self):
        newStateValue = np.zeros((self.max_cars+1,self.max_cars+1))
        for i, j in self.states:
            newStateValue[i, j] = self.expectedReturn([i, j], self.policy[i, j])
        self.diff = np.sum(np.abs(newStateValue - self.stateValue))
        if self.diff < self.theta:
            self.stateValue[:] = newStateValue
            self.improvePolicy = True
        self.stateValue[:] = newStateValue  
        
    def expectedReturn(self,state,action):
        returns = 0.0
        
        returns -=self.move_car_cost*abs(action)
        
        avalibleOfFirstLoc = int(min(state[0]-action,self.max_cars))
        avalibleOfSecondLoc = int(min(state[1]+action,self.max_cars))
        
        for rentalRequestFirstLoc in range(0,avalibleOfFirstLoc+1):
            for rentalRenquestSecondLoc in range(0,avalibleOfSecondLoc+1):
                
                numOfCarsFirstLoc = avalibleOfFirstLoc
                numOfCarsSecondLoc = avalibleOfSecondLoc
               
                realRentalFirstLoc = min(numOfCarsFirstLoc,rentalRequestFirstLoc)
                realRentalSecondLoc = min(numOfCarsSecondLoc,rentalRenquestSecondLoc)
                
                reward = (realRentalFirstLoc+realRentalSecondLoc)*self.rental_credit
                
                numOfCarsFirstLoc -=realRentalFirstLoc
                numOfCarsSecondLoc -=realRentalSecondLoc
                
                prob = poission(rentalRequestFirstLoc,self.rental_request_first_loc)*poission(realRentalSecondLoc,self.rental_request_second_loc)
                
                #当天借车当天还Ϊtrue
                constantReturndCars = True
                if(constantReturndCars):
                    returnedCarsFirstLoc = self.returns_first_loc
                    returnedCarsSecondLoc = self.returns_second_loc
                    numOfCarsFirstLoc = min(numOfCarsFirstLoc + returnedCarsFirstLoc, self.max_cars)
                    numOfCarsSecondLoc = min(numOfCarsSecondLoc + returnedCarsSecondLoc, self.max_cars)
                    returns += prob * (reward + self.discount * self.stateValue[numOfCarsFirstLoc, numOfCarsSecondLoc])
                else:
                    numOfCarsFirstLoc_ = numOfCarsFirstLoc
                    numOfCarsSecondLoc_ = numOfCarsSecondLoc
                    prob_ = prob
                    for returnedCarsFirstLoc in range(0, realRentalFirstLoc+realRentalSecondLoc+1):
                        for returnedCarsSecondLoc in range(0, realRentalFirstLoc+realRentalSecondLoc+1):
                            numOfCarsFirstLoc = numOfCarsFirstLoc_
                            numOfCarsSecondLoc = numOfCarsSecondLoc_
                            prob = prob_
                            numOfCarsFirstLoc = min(numOfCarsFirstLoc + returnedCarsFirstLoc, self.max_cars)
                            numOfCarsSecondLoc = min(numOfCarsSecondLoc + returnedCarsSecondLoc, self.max_cars)
                            prob = poission(returnedCarsFirstLoc, self.returns_first_loc) * \
                                 poission(returnedCarsSecondLoc, self.returns_second_loc) * prob
                            returns += prob * (reward + self.discount * self.stateValue[numOfCarsFirstLoc,numOfCarsSecondLoc])
        return returns
                
    def policyIteration(self):
        while True:
            if self.improvePolicy:
                self.improvement()
            if self.endImprove:
                break;
            
            self.evaluation()
    
    def asynPolicyIteration(self):
        pass
    
    
    def valueIteration(self):
        self.diff = 100
        while self.diff>self.theta:
            self.diff = 0.0
            
            oldStateValue = np.zeros((self.max_cars+1,self.max_cars+1))
            oldStateValue[:] = self.stateValue[:]
            oldPolicy = np.zeros((self.max_cars+1,self.max_cars+1))
            oldPolicy[:] = self.policy[:]
            
            for i, j in self.states:
                actionReturns = []
                for action in self.actions:
                    if(action>=0 and i>=action) or(action<0 and j>=abs(action)):
                        returns = self.expectedReturn([i,j],action)
                        actionReturns.append(returns)
                    else:
                        actionReturns.append(-float('inf'))
                bestAction = np.argmax(actionReturns)
                self.policy[i,j] = self.actions[bestAction]  
                self.stateValue[i, j] = np.max(actionReturns)
            
            self.diff = np.sum(np.abs(self.stateValue - oldStateValue))
            self.policyDiff = np.sum(oldPolicy!=self.policy) 
            self.valueImprovementInd += 1
            print('changeCount..',self.valueImprovementInd)
            print ('policy for',np.sum(self.policy!=oldPolicy),'states changes') 
            print ('diff..',self.diff)
 
 
    def prettyPrint(self,data,labels):
        global figureIndex
        fig = plt.figure(figureIndex)
        figureIndex +=1
        ax = fig.add_subplot(111,projection='3d')
        AxisXPrint = []
        AxisYPrint = []
        AxisZ = []
        for i,j in self.states:
            AxisXPrint.append(i)
            AxisYPrint.append(j)
            AxisZ.append(data[i,j])
            
        ax.scatter(AxisXPrint,AxisYPrint,AxisZ)
        ax.set_xlabel(labels[0])   
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.set_title(labels[3])
        
    def prettyPrintPolicy(self,title):
        self.prettyPrint(self.policy,['#of cars in first location','#of cars in second location','of cars to move during night',title+'Action'])
        
    def prettyPrintStateValue(self,title):
        self.prettyPrint(self.stateValue,['# of cars in first location', '# of cars in second location', 'expected returns',title+'ExpectedReturn'])
        
        
print ('start policy Iteration =====================================')
startTime = time.time()
cars = JackCar()
cars.policyIteration()
cars.prettyPrintPolicy('Policy Iteration:')
cars.prettyPrintStateValue('Policy Iteration:')
endTime = time.time()
  
print ('Policy Iteration Time:',endTime-startTime)
print ('Policy Iteration Count:',cars.policyImprovementInd)
print ('Policy ITeration Diff:',cars.diff)
  
print()
print()
print('start Value Iteration=========================================')
startTime = time.time()
carsValue = JackCar()
carsValue.valueIteration()
carsValue.prettyPrintPolicy('Value Iteration:')
carsValue.prettyPrintStateValue('Value Iteration:')
endTime = time.time()
   
print ('Value Iteration Time:',endTime-startTime)
print ('Value Iteration count:',carsValue.valueImprovementInd)
print('Value Iteration Diff:',carsValue.diff)
 

 
#画图，将对比结果画在一张图上
def printConparePicture(states,dictPolicys,dictExpectedReturns,title,CompareArr):
        #figureIndex +=1
        fig, axs= plt.subplots(3, 2, figsize=plt.figaspect(0.5), subplot_kw={'projection': '3d'})
        count  = 0
        for i in range(0,3):
            for j in range(0,2):
                if(count>=len(dictPolicys)):
                    break;
                AxisXPrint = []
                AxisYPrint = []
                AxisZ = []
                for m,n in states:
                    AxisXPrint.append(m)
                    AxisYPrint.append(n)
                    AxisZ.append(dictPolicys[count][i,j])
                axs[i,j].scatter(AxisXPrint,AxisYPrint,dictPolicys[count])
                axs[i,j].set_xlabel('#first location')   
                axs[i,j].set_ylabel('#in second location')
                axs[i,j].set_zlabel('#Actions')
                axs[i,j].set_title(title+str(CompareArr[count]))
                count += 1
  
  
        fig, axs= plt.subplots(3,2, figsize=plt.figaspect(0.5), subplot_kw={'projection': '3d'})
        count  = 0
        for i in range(0,3):
            for j in range(0,2):
                if(count>=len(dictExpectedReturns)):
                    break;
                AxisXPrint = []
                AxisYPrint = []
                AxisZ = []
                for m,n in states:
                    AxisXPrint.append(m)
                    AxisYPrint.append(n)
                    AxisZ.append(dictExpectedReturns[count][i,j])
                axs[i,j].scatter(AxisXPrint,AxisYPrint,dictExpectedReturns[count])
                axs[i,j].set_xlabel('#first location')   
                axs[i,j].set_ylabel('#in second location')
                axs[i,j].set_zlabel('#Expected Return')
                axs[i,j].set_title(title+str(CompareArr[count]))
                count += 1                
          
# states = [] 
# Times=[]
# print ('start compare that different discount  to affect the outcome==========================')
# DiscountChangePolicys = []
# DiscountChangeExpectedReturns = []
# for discount in range(0,11):
#     startTime = time.time()
#     cars = JackCar(disCount=discount*0.1)
#     cars.policyIteration()
#     print("========================start with discount = ",discount*0.1,'==============================')
#     DiscountChangePolicys.append(cars.policy)
#     DiscountChangeExpectedReturns.append(cars.stateValue)
#     states = cars.states
#     endTime = time.time()
#     Times.append(endTime-startTime)
#   
# printConparePicture(states,DiscountChangePolicys,DiscountChangeExpectedReturns,"DisCount=",0.1)
#  
# plt.figure(figureIndex)
# plt.plot(range(0,11),Times);
# plt.xlabel("#discount*10")
# plt.ylabel("#speed Time")
# plt.title("compare result with change discount")


print ('start compare that different theta  to affect the outcome==========================')
ThetaChangePolicys = []
TheTaChangeExpectedReturns = []
ThetaChangeDiff = []
ThetaChangePolicyDiff = []
CompareArr = [10,0.1,0.01,0.001,0.0001,0]
Times=[]
PolicyImprovmentCount = []

for theta in CompareArr:
    startTime = time.time()
    cars = JackCar(theta = theta)
    cars.valueIteration()
    print(("=============================start with theta = ",theta,"============================="))
    ThetaChangePolicys.append(cars.policy)
    TheTaChangeExpectedReturns.append(cars.stateValue)
    ThetaChangeDiff.append(cars.diff)
    ThetaChangePolicyDiff.append(cars.policyDiff)
    states = cars.states
    endTime = time.time()
    Times.append(endTime-startTime)
    PolicyImprovmentCount.append(cars.valueImprovementInd)
# 
printConparePicture(states,ThetaChangePolicys,TheTaChangeExpectedReturns,"Theta=",CompareArr)
# 
# 
# plt.figure(figureIndex)
# plt.plot(range(0,5),Times);
# plt.xlabel("#discount*10")
# plt.ylabel("#speed Time")
# plt.title("compare result with change discount")

print('policy Improvement count',PolicyImprovmentCount)
print('speed time:',Times)
print ('the stateValue diff between last two iteration:',ThetaChangeDiff)
print('the policy differents between last two iteration:',ThetaChangePolicyDiff)

plt.show()

