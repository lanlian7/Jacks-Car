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
    def __init__(self):
        self.max_cars = 20
        self.max_move_of_cars = 5
        self.rental_request_first_loc = 3
        self.rental_request_second_loc = 4
        self.returns_first_loc = 3
        self.returns_second_loc = 2
        
        self.discount = 0.9
        
        self.rental_credit = 10
        self.move_car_cost = 2
        
        self.improvePolicy = False
        self.policyImprovementInd = 0
        self.endImprove = False
        
        #An up bound for poisson distribution
        #if n is greater than this value ,then the probability of getting n is truncated to 0
        self.poisson_up_bound = 11
        
        #�ƽ���ȷ��
        self.theta = 0.001
        
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
        if np.sum(np.abs(newStateValue - self.stateValue)) < self.theta:
            self.stateValue[:] = newStateValue
            self.improvePolicy = True
        self.stateValue[:] = newStateValue  
        
    def expectedReturn(self,state,action):
        returns = 0.0
        
        returns -=self.move_car_cost*abs(action)
        
        for rentalRequestFirstLoc in range(0,self.poisson_up_bound):
            for rentalRenquestSecondLoc in range(0,self.poisson_up_bound):
                
                numOfCarsFirstLoc = int(min(state[0]-action,self.max_cars))
                numOfCarsSecondLoc = int(min(state[1]+action,self.max_cars))
                
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
                    for returnedCarsFirstLoc in range(0, self.poisson_up_bound):
                        for returnedCarsSecondLoc in range(0, self.poisson_up_bound):
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
        
    def prettyPrintPolicy(self):
        self.prettyPrint(self.policy,['#of cars in first location','#of cars in second location','#of cars to move during night'])
        
    def prettyPrintStateValue(self):
        self.prettyPrint(self.stateValue,['# of cars in first location', '# of cars in second location', 'expected returns'])
    
cars = JackCar()
cars.policyIteration()
cars.prettyPrintPolicy()
cars.prettyPrintStateValue()

plt.show()

