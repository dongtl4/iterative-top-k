# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 10:43:01 2024

local databases

@author: dongh
"""

import numpy as np
# import csv
import pandas as pd
# import time
# from machine import analysis_center

class node:    
    def __init__(self, scores, indexes, mean_ping=20, upload=45, download=45, stored=False, node_name=''):
        """
        create database with new data and generate random variables
        """
        self.data = pd.Series(data=scores, index=indexes)
        self.darray = []
        self.dindex = []
        self.sent = np.zeros(len(self.data), dtype='bool') # only used when needed to save the sent position
        self.random = pd.Series([], name='random sample', dtype = 'float64')
        self.threshold = 0
        self.prevthres = np.inf
        self.fullindex = 0
        # performance tracking
        self.ID_sent = {} # number of item's ID sent
        self.score_sent = {} # number of local score sent
        self.rd_value_sent = {} # number of random value sent
        self.compute_time = {}
        self.extended = {}
        self.nbrounds = 0 # number of round-trip communications
        self.received_message = {}
        self.ping = mean_ping #ms
        self.upspeed = upload #Mbps
        self.downspeed = download #Mbps
        self.cursor = 0 #for sorted access
        
        
    def sort(self, t=1):
        if t==1:
            self.data = self.data.sort_values(ascending=False)
        elif t==2:
            self.random = self.random.sort_values(ascending=False)
        
    def generate_bin_rand(self, origin, p, replace=False):
        """
        generate random variables that follow binomial distribution
        """
        if replace:
            self.random = pd.Series(np.random.binomial(origin, p), origin.index, dtype=self.data.dtypes)
        else:
            self.random = pd.concat([self.random, pd.Series(np.random.binomial(origin, p), index=origin.index, dtype=self.data.dtypes)])
        
    def generate_poisson_rand(self, origin, index, p, replace=False):
        """
        generate random variables that follow Poisson distribution
        """
        if replace:
            self.random = pd.Series(np.random.poisson(origin*p), index=index, dtype=self.data.dtypes)
        else:
            self.random = pd.concat([self.random, pd.Series(np.random.poisson(origin*p), index=origin.index, dtype=self.data.dtypes)])
     
    def generate_exp_rand(self, origin, index, replace=False):
        """
        generate random variables that follow exponential distribution
        """
        if replace:
            self.random = pd.Series(np.random.exponential(1/origin), index=index, dtype=self.data.dtypes)
        else:
            self.random = pd.concat([self.random, pd.Series(np.random.exponential(1/origin), index=index, dtype=self.data.dtypes)])
    
    def dup_data(self):
        self.darray = np.array(self.data.values)
        self.dindex = np.array(self.data.index)
        self.data = pd.Series(dtype='float64')
        
    def sub_data(self):
        self.data = pd.Series(data=self.darray, index=self.dindex)
        self.darray = []
        self.dindex = []
        
    # clear all except data
    def refresh(self):
        self.threshold = 0
        self.prevthres = np.inf
        self.random = pd.Series([], name='random sample', dtype = 'float64')
        self.sent = np.zeros(max(len(self.data), len(self.darray)), dtype='bool')
        # performance tracking
        self.compute_time = {}
        self.ID_sent = {} # number of item's ID sent
        self.score_sent = {} # number of local score sent
        self.rd_value_sent = {} # number of random value sent
        self.extended = {}
        self.nbrounds = 0 # number of round-trip communications
        self.received_message = {}
    
    # following func will return the list of collected items/samples by specific condition
    # collecting by list of ID
    def collect_ID(self, origin, collected_list=[]):
        valid_indices = np.intersect1d(collected_list, origin.index)
        return origin[valid_indices]
        
    # collecting item top down by values
    def collect_highest(self, origin, number_of_items):
        considered = origin.nlargest(number_of_items).index
        return origin[considered]
    
    # collecting by threshold(s)
    def collect_threshold(self, origin, lower=-np.inf, upper=np.inf):
        return origin[(origin >= lower) & (origin < upper)]
    
    # for descending order
    def sorted_collect_threshold(self, origin, lower=-np.inf, upper=np.inf):
        l = len(origin)
        up = l-np.searchsorted(origin[::-1], upper)
        low = l-np.searchsorted(origin[::-1], lower)
        return origin[up:low]
    
    def ul_sorted_collect_threshold(self, origin, lower=-np.inf, upper=np.inf):
        l = len(origin)
        up = l-np.searchsorted(origin[::-1], upper)
        low = l-np.searchsorted(origin[::-1], lower)
        return up, low
        
    # other function
    def histogram(self, bins):
        pass
    
    def histobloom(self, bins, c):
        pass