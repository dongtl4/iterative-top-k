# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 10:40:04 2024

Analysis center

@author: dongh
"""
# from machine import local_databases
import numpy as np
import pandas as pd

class middleware:
    tem_list = [] # temporary list of items' ID send to middleware, collected from received message
    items_gathered = pd.Series(dtype='int64') # list of items gathered and their value (Q)
    rm = [] # received message
    
    def __init__(self, S, p):
        self.S = S
        self.p = p
    
    # boardcast message to local databases
    def boardcast(self, message, local_list):
        for database in local_list:
            database.rm.append(message)
    
    # boardcast message to estimate total sum of all item's score
    def boardcast_estimate_S(self, local_list):
        self.boardcast(['sum',-1], local_list)
    
    # boardcast message to estimate the score of 1 item 
    # that correspond with tickets remained
    def boardcast_estimate_Si(self, _item, local_list):
        self.boardcast(['sum', _item], local_list)
        
    # calculate the sum of message received corresponding with type boardcast 'sum'
    def calculate(self):
        result = 0
        for i in self.rm:
            result = result + i
        return result
    
    # boardcast its p to local databases
    def boardcast_p(self, local_list):
        self.boardcast([self.p], local_list)
        
    # collect items' ID that not in items_gathered to put into tem_list
    # using when rm is a list of items' ID list
    def collect(self):
        for i in self.rm:
            for j in i:
                if j not in self.items_gathered.index:
                    if j not in self.tem_list:
                        self.tem_list.append(j)
    
    # drop all message received
    def empty_message(self):
        self.rm = []
        
    # drop all ticket list
    def empty_ticket(self):
        self.ticket_list = []
    