#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:52:51 2024
For convenient, the index in computing is np.arange(n)

@author: vandong
"""

# from machine import local_nodes as node
import numpy as np
import pandas as pd
from rbloom import Bloom
import scipy
import time
import multiprocessing
# import gc  

def TPUT(nodes, n, k, a=1, approx=False, adapt=False):
    """
    Three phase Uniform threshold
    """
    # phase 1
    hdict = {}
    temp_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        if adapt:
            start = time.time()
            bins = np.percentile(nodes[i].darray, np.linspace(0, 100, 100 + 1))
            hist = np.histogram(nodes[i].darray, bins=bins)
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] += end-start
            hdict.update({i:hist})
            # save the size (in bytes) used by the histogram
            nodes[i].extended[nodes[i].nbrounds] = 100+101*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    tau1=np.partition(temp_score, -k)[-k]*a
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    if adapt:
        start = time.time()
        hist = pd.DataFrame(np.zeros((100, len(nodes))))
        for i in range(len(nodes)):
            hist.iloc[0, i] = hdict[i][0][-1]
            for j in range(1, 100):
                hist.iloc[j, i] = sum(hdict[i][0][-j-1:])
        kmin = tau1/a
        # using binary process to find appropriate threshold
        temp=2*kmin
        tempup = 0
        cnumu = n
        cnumd = 0
        cnum = np.ceil((cnumu + cnumd)/2)
        while (temp > a*kmin) or (tempup < a*kmin):
            temp = 0
            tempup = 0
            cnum = np.ceil((cnumu + cnumd)/2)
            for i in range(len(nodes)):
                ind = hist.iloc[:, i][hist.iloc[:, i] <= cnum]
                if len(ind)==0:
                    ind = -1
                else:
                    ind = ind.index[-1]
                if ind == 99:
                    temp+=0
                    tempup+=hdict[i][1][-ind-2]
                else:
                    temp+=hdict[i][1][-ind-3]
                    tempup+=hdict[i][1][-ind-2]
            if temp > a*kmin:
                cnumd = cnum
            if tempup < a*kmin:
                cnumu = cnum
        for i in range(len(nodes)):
            ind = hist.iloc[:, i][hist.iloc[:, i] <= cnum]
            if len(ind)==0:
                ind = -1
            else:
                ind = ind.index[-1]
            if ind == 99:
                nodes[i].threshold = 0 + hdict[i][1][-ind-2]/tempup*(kmin*a-temp)
            else:
                nodes[i].threshold = hdict[i][1][-ind-3] + hdict[i][1][-ind-2]/tempup*(kmin*a-temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        for i in range(len(nodes)):
            nodes[i].threshold = tau1/len(nodes)
            
    # phase 2
    miss = np.ones(n)*tau1
    message2=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8 # send the threshold to each local nodes
        start=time.time()
        nodes[i].cursor = len(nodes[i].darray)-np.searchsorted(nodes[i].darray[::-1], nodes[i].threshold)
        ind = [nodes[i].dindex[k:nodes[i].cursor], nodes[i].darray[k:nodes[i].cursor]]
        end=time.time()
        message2.append(ind)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind[0])
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message2[i][0]]+=message2[i][1]
        miss[message[i][0]]-=nodes[i].threshold
        miss[message2[i][0]]-=nodes[i].threshold
    tau2 = np.partition(temp_score, -k)[-k]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    if approx:
        start=time.time()
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        # avoid missing items in final due to very small error happened when calculating
        start=time.time()
        best = temp_score + miss +1e-5
        final = np.where(best>tau2)[0]
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    
    if not approx:
        message3=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
            message3.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message3[i][0]]+=message3[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start 
    return collected, cct

def FRUT(nodes, n, k, a=0.5, approx=False, adapt=False):
    """
    Four Rounds Uniform Threshold
    """
    # phase 1
    hdict = {}
    temp_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        if adapt:
            start = time.time()
            bins = np.percentile(nodes[i].darray, np.linspace(0, 100, 100 + 1))
            hist = np.histogram(nodes[i].darray, bins=bins)
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] += end-start
            hdict.update({i:hist})
            # save the size (in bytes) used by the histogram
            nodes[i].extended[nodes[i].nbrounds] = 100+101*8
    start = time.time()
    D=set()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
        D=D.union(set(message[i][0]))
    D=list(D)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    message1=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(D)*8
        start = time.time()
        ind = np.where(np.isin(nodes[i].dindex[k:], D))[0]
        message1.append([nodes[i].dindex[k:][ind], nodes[i].darray[k:][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message1[i][0]]+=message1[i][1]
    tau1=np.partition(temp_score, -k)[-k]*a
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    if adapt:
        start = time.time()
        hist = pd.DataFrame(np.zeros((100, len(nodes))))
        for i in range(len(nodes)):
            hist.iloc[0, i] = hdict[i][0][-1]
            for j in range(1, 100):
                hist.iloc[j, i] = sum(hdict[i][0][-j-1:])
        kmin = tau1/a
        # using binary process to find appropriate threshold
        temp=2*kmin
        tempup = 0
        cnumu = n
        cnumd = 0
        cnum = np.ceil((cnumu + cnumd)/2)
        while (temp > a*kmin) or (tempup < a*kmin):
            temp = 0
            tempup = 0
            cnum = np.ceil((cnumu + cnumd)/2)
            for i in range(len(nodes)):
                ind = hist.iloc[:, i][hist.iloc[:, i] < cnum]
                if len(ind)==0:
                    ind = -1
                else:
                    ind = ind.index[-1]
                if ind == 99:
                    temp+=0
                    tempup+=hdict[i][1][-ind-2]
                else:
                    temp+=hdict[i][1][-ind-3]
                    tempup+=hdict[i][1][-ind-2]
            if temp > a*kmin:
                cnumd = cnum
            if tempup < a*kmin:
                cnumu = cnum
        for i in range(len(nodes)):
            ind = hist.iloc[:, i][hist.iloc[:, i] < cnum]
            if len(ind)==0:
                ind = -1
            else:
                ind = ind.index[-1]
            if ind == 99:
                nodes[i].threshold = 0 + hdict[i][1][-ind-2]/tempup*(kmin*a-temp)
            else:
                nodes[i].threshold = hdict[i][1][-ind-3] + hdict[i][1][-ind-2]/tempup*(kmin*a-temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        for i in range(len(nodes)):
            nodes[i].threshold = tau1/len(nodes)
            
    # phase 2
    miss = np.ones(n)*tau1
    message2=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8 # send the threshold to each local nodes
        start=time.time()
        nodes[i].cursor = len(nodes[i].darray)-np.searchsorted(nodes[i].darray[::-1], nodes[i].threshold)
        ind = np.where(~np.isin(nodes[i].dindex[k:nodes[i].cursor], D))[0]
        message2.append([nodes[i].dindex[k:nodes[i].cursor][ind], nodes[i].darray[k:nodes[i].cursor][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message2[i][0]]+=message2[i][1]
        miss[message2[i][0]]-=nodes[i].threshold
    miss[D]-=tau1
    tau2 = np.partition(temp_score, -k)[-k]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    if approx:
        start=time.time()
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        # avoid missing items in final due to very small error happened when calculating
        start=time.time()
        best = temp_score + miss +1e-5
        final = np.where(best>tau2)[0]
        cfinal = list(set(final)-set(D))
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    
    # phase 3
    if not approx:
        message3=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], cfinal))[0]            
            message3.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message3[i][0]]+=message3[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
    return collected, cct

def evaluate(alg, nodes, k, output, cct, gs, centernet = 10e9, epsilon=0, refresh=True):
    """
    Evaluating the result
    """
    # accuracy
    er = len(set(gs[:k].index)-set(output[:k].index))
    temp = gs[:k].values - gs[output.index[:k]].values*(1+epsilon)
    ere = len(temp[temp > 1e-5])            
    mse =  max(gs[:k].values/gs[output.index[:k]].values-1)
    ase = sum(gs[:k].values/gs[output.index[:k].values]-1)/k
    # # bandwidth
    IDs = sum([sum(i.ID_sent.values()) for i in nodes])
    extd = sum([sum(i.extended.values()) for i in nodes])
    scs = sum([sum(i.score_sent.values()) for i in nodes])
    bcm = sum(nodes[0].received_message.values())
    mlb = max([sum(i.ID_sent.values())*8 + sum(i.score_sent.values())*8 + sum(i.extended.values()) for i in nodes])
    nbr = max(i.nbrounds for i in nodes)
    # compute time
    mct = max([sum(x.compute_time.values()) for x in nodes])
    sct = sum(cct.values())
    # total time
    round_time = {}
    for i in range(1,nodes[0].nbrounds+1):
        round_time[i] = max([node.compute_time.get(i, 0) for node in nodes]) # compute time
        round_time[i] += max([node.received_message.get(i, 0)*544/514/node.downspeed/1e6 for node in nodes]) # receiving messages time
        round_time[i] += max(max([(node.ID_sent.get(i, 0)*8+node.extended.get(i,0)+node.score_sent.get(i, 0)*8)*544/514*8/node.upspeed/1e6 for node in nodes]), sum([(node.ID_sent.get(i, 0)*8+node.extended.get(i,0)+node.score_sent.get(i, 0)*8)*544/514*8 for node in nodes])/centernet) # upload information time
        round_time[i] += max([np.random.poisson(100)/100*node.ping*2/1000 for node in nodes]) # ping
    # To have a simple code and be able to run multiple algorithms parallelly, I using single thread numpy for all calculation of the query initiator
    # The assumption here is that the query center have 8x(CPU & RAM). Thus the 'cct' is divided by 8.
    final = sum(cct.values())/8 + sum(round_time.values()) 
    if refresh:
        for i in nodes:
            i.refresh()
    return [alg, er, ere, mse, ase, IDs, extd, scs, bcm, mlb, nbr, mct, sct, final]

def Sstar(p, epsilon, delta):
    """
    S* function described in "Boaz Patt-Shamir and Allon Shafrir. 2008. 
    Approximate distributed top-k queries. Distributed Computing"
    """
    r1 = np.log(1+epsilon) - np.log(p*delta) + 4
    r2 = (1+epsilon)*np.log(1+epsilon)
    r3 = epsilon*np.log(epsilon/np.log(1+epsilon)) + np.log(1+epsilon) - epsilon
    return (r1/p)*(r2/r3)

def samplingTopk(nodes, n, k, delta=0.05, epsilon=1):
    """
    Sampling based top frequent items
    """
    estimated_score=pd.Series(0, index=nodes[0].fullindex, dtype='float64')
    S = []
    s = 0
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 1
        start = time.time()
        s += sum(nodes[i].darray)
        end = time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
    S.append(s)
    p = 2/k
    l = 0
    loopcont = True
    cct={}
    
    while loopcont:
        l+=1
        temp_score=np.zeros(n)
        s = S[0] - sum(estimated_score)
        S.append(s)
        p = p/2*(S[l-1]/S[l])
        pstar = Sstar(p, epsilon, delta)/S[l]
        message=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = 8
            start = time.time()
            sample = pd.Series(np.random.binomial(nodes[i].darray.astype(int), pstar), index=nodes[i].dindex)
            sample = sample[sample>0]
            message.append(sample)
            end = time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(sample)
            nodes[i].score_sent[nodes[i].nbrounds] = len(sample)
        start = time.time()
        for i in range(len(nodes)):
            temp_score[message[i].index]+=message[i].values
        temp_score = pd.Series(temp_score)
        # estimation part (inspired by the algorithm PAC of Hubschle-Schneider et al 2016, not applied directly bcs PAC consider score error is a proportion of S, while the accuracy goal is proportion of scores in T(k, E))
        threshold = p*Sstar(p, epsilon, delta)/(1+epsilon)
        temp_score=pd.Series(temp_score)
        T = temp_score[temp_score >= threshold].index
        estimated_score[T] = temp_score[T]/pstar
        estimated_score.sort_values(ascending=False, inplace=True)
        if len(estimated_score[estimated_score>0]) >=k:
            # here change the stopping criteria: p*S[l]/S[1] < 1/4S*min(T_(k,Q)) to p*S[l]/S[1] < (1+epsilon)/S*min(T_(k,Q)), which made the algorithm faster at least 2 times
            loopcont = p*S[l]/S[1] > estimated_score.iloc[k-1]/(S[0])*(1+epsilon)
            # to be honest, idk what is the meaning of this condition, just put it in according the paper
            r = min(estimated_score.loc[list(T)])/(S[0])
            q = (1+epsilon/2)*estimated_score.iloc[k-1]/(S[0])
            if q > r:
                alpha = q/r
                if np.exp(-len(T)*q*(alpha-1)*(alpha-1)/(alpha*alpha)) < q*delta/5:
                    loopcont = False
        end = time.time()
        cct[nodes[0].nbrounds]=end-start
    return estimated_score, cct

def ITT(nodes, n, k, c=2, a=2/3, approx=False, adapt=False):
    """
    Iterative Threshold Top-k
    """
    threshold = np.zeros(len(nodes))
    temp_score = np.zeros(n)
    best_score = np.ones(n)
    cct = {}
    hdict = {}
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        nodes[i].cursor=k
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].sent[:k] = True
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        if adapt:
            start = time.time()
            bins = np.percentile(nodes[i].darray, np.linspace(0, 100, 100 + 1))
            hist = np.histogram(nodes[i].darray, bins=bins)
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] += end-start
            hdict.update({i:hist})
            nodes[i].extended[nodes[i].nbrounds] = 100+101*8
    start = time.time()
    for i in range(len(nodes)):
        threshold[i] = min(message[i][1])   
    t = sum(threshold)
    best_score *= t
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
        best_score[message[i][0]]+=(message[i][1] - threshold[i])
    if not adapt:
        miss = np.ones(n)*len(nodes)
        for i in range(len(nodes)):
            miss[message[i][0]]-=1
    p=np.partition(temp_score, -k)[-k]
    b=np.partition(best_score, -k)[-k]
    if adapt:
        hist = pd.DataFrame(np.zeros((100, len(nodes))))
        for i in range(len(nodes)):
            hist.iloc[0, i] = hdict[i][0][-1]
            for j in range(1, 100):
                hist.iloc[j, i] = sum(hdict[i][0][-j-1:])
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    while p+1e-5 < b:
        b = max(b/2, p)
        if adapt:
            start = time.time()
            kmin = b*a
            # using binary process to find appropriate threshold
            temp=2*kmin
            tempup = 0
            cnumu = n
            cnumd = 0
            cnum = (cnumu + cnumd)/2
            while (temp > kmin) or (tempup < kmin):
                temp = 0
                tempup = 0
                cnum=np.ceil((cnumu + cnumd)/2)
                for i in range(len(nodes)):
                    ind = hist.iloc[:, i][hist.iloc[:, i] < cnum]
                    if len(ind)==0:
                        ind = -1
                    else:
                        ind = ind.index[-1]
                    if ind == 99:
                        temp+=0
                        tempup+=hdict[i][1][-ind-2]
                    else:
                        temp+=hdict[i][1][-ind-3]
                        tempup+=hdict[i][1][-ind-2]
                if temp > kmin:
                    cnumd = cnum
                if tempup < kmin:
                    cnumu = cnum
            for i in range(len(nodes)):
                ind = hist.iloc[:, i][hist.iloc[:, i] < cnum]
                if len(ind)==0:
                    ind = -1
                else:
                    ind = ind.index[-1]
                if ind == 99:
                    nodes[i].threshold = 0 + hdict[i][1][-ind-2]/tempup*(kmin-temp)
                else:
                    nodes[i].threshold = hdict[i][1][-ind-3] + hdict[i][1][-ind-2]/tempup*(kmin-temp)
            end=time.time()
            cct[nodes[0].nbrounds]+=end-start
        else:
            for i in range(len(nodes)):
                nodes[i].threshold = b*a/len(nodes)
        message=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = 8 # send the threshold to each local nodes
            start=time.time()
            next_cursor = len(nodes[i].darray)-np.searchsorted(nodes[i].darray[::-1], nodes[i].threshold)
            ind = [nodes[i].dindex[nodes[i].cursor:next_cursor], nodes[i].darray[nodes[i].cursor:next_cursor]]
            end=time.time()
            message.append(ind)
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind[0])
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind[0])
            nodes[i].cursor = max(next_cursor, nodes[i].cursor)
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message[i][0]]+=message[i][1]
            if not adapt:
                miss[message[i][0]]-=1
        if adapt:
            miss = np.ones(n)*sum([node.threshold for node in nodes])
            for i in range(len(nodes)):
                miss[nodes[i].dindex[:nodes[i].cursor]]-=nodes[i].threshold
            best = miss+temp_score+1e-5
            b = min(b, np.partition(best, -k)[-k])
        if not adapt:
            best = miss*nodes[0].threshold+temp_score
            b = min(b, np.partition(best, -k)[-k])
        p=np.partition(temp_score, -k)[-k]
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
    
    if approx:
        start=time.time()
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        start=time.time()
        final = np.where(best>p)[0]
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
        message=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
            message.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message[i][0]]+=message[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start 
    return collected, cct
        

def IES(nodes, n, k, delta=0.05, epsilon=0.5, approx=True, adapt=False):
    """
    Iterative Exponential Sampling
    """
    threshold = np.zeros(len(nodes))
    temp_score = np.zeros(n)
    best_score = np.ones(n)
    Q=[]
    Filter=[]
    cct={}
    message=[]
    hdict={}
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        nodes[i].cursor=k
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].sent[:k] = True
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        if adapt:
            start = time.time()
            bins = np.percentile(nodes[i].darray, np.linspace(0, 100, 100 + 1))
            hist = np.histogram(nodes[i].darray, bins=bins)
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] += end-start
            hdict.update({i:hist})
            # save the size (in bytes) used by the histogram
            nodes[i].extended[nodes[i].nbrounds] = 100+101*8
    start = time.time()
    for i in range(len(nodes)):
        threshold[i] = min(message[i][1])   
    t = sum(threshold)
    best_score *= t
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
        best_score[message[i][0]]+=(message[i][1] - threshold[i])
    if not adapt:
        miss = np.ones(n)*len(nodes)
        for i in range(len(nodes)):
            miss[message[i][0]]-=1
    p=np.partition(temp_score, -k)[-k]
    b=np.partition(best_score, -k)[-k]
    if adapt:
        hist = pd.DataFrame(np.zeros((100, len(nodes))))
        for i in range(len(nodes)):
            hist.iloc[0, i] = hdict[i][0][-1]
            for j in range(1, 100):
                hist.iloc[j, i] = sum(hdict[i][0][-j-1:])
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    while p+1e-5 < b:
        b = max(b/2, p)
        if adapt:
            start = time.time()
            kmin = b*epsilon
            # using binary process to find appropriate threshold
            temp=2*kmin
            tempup = 0
            cnumu = n
            cnumd = 0
            cnum = (cnumu + cnumd)/2
            while (temp > kmin) or (tempup < kmin):
                temp = 0
                tempup = 0
                cnum=np.ceil((cnumu + cnumd)/2)
                for i in range(len(nodes)):
                    ind = hist.iloc[:, i][hist.iloc[:, i] < cnum]
                    if len(ind)==0:
                        ind = -1
                    else:
                        ind = ind.index[-1]
                    if ind == 99:
                        temp+=0
                        tempup+=hdict[i][1][-ind-2]
                    else:
                        temp+=hdict[i][1][-ind-3]
                        tempup+=hdict[i][1][-ind-2]
                if temp > kmin:
                    cnumd = cnum
                if tempup < kmin:
                    cnumu = cnum
            for i in range(len(nodes)):
                ind = hist.iloc[:, i][hist.iloc[:, i] < cnum]
                if len(ind)==0:
                    ind = -1
                else:
                    ind = ind.index[-1]
                if ind == 99:
                    nodes[i].threshold = 0 + hdict[i][1][-ind-2]/tempup*(kmin-temp)
                else:
                    nodes[i].threshold = hdict[i][1][-ind-3] + hdict[i][1][-ind-2]/tempup*(kmin-temp)
            end=time.time()
            cct[nodes[0].nbrounds]+=end-start
        else:
            for i in range(len(nodes)):
                nodes[i].threshold = b*epsilon/len(nodes)
        Q_temp = set()
        message=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds+=1
            nodes[i].received_message[nodes[i].nbrounds] = 8
            # generate exponential random sample in range t2 <= s_ij < t1
            start = time.time()
            u, l = nodes[i].ul_sorted_collect_threshold(nodes[i].darray, lower=nodes[i].threshold, upper=nodes[i].prevthres)
            nodes[i].generate_exp_rand(nodes[i].darray[u:l], nodes[i].dindex[u:l], replace=True)
            # collect candidates
            candidates = nodes[i].collect_threshold(nodes[i].random, upper=-np.log(delta)/p).index
            message.append(candidates)
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].prevthres = nodes[i].threshold
            nodes[i].ID_sent[nodes[i].nbrounds] = len(candidates)
        start = time.time()
        for i in range(len(nodes)):
            for x in message[i]:
                Q_temp.add(x)
        Q_temp = Q_temp - set(Q)
        if len(Q) < k:
            Q = Q + list(Q_temp)
        if len(Q) >= k:
            if len(Filter)==0: # which indicate this is the first time len(Q) >= k, no Filter is created yet
                Filter.append(Bloom(len(Q)+10, 0.01))
                Filter[-1].update(Q)
            else: # if there is filter(s) in Filter, so only need to turn ids in Q_temp to a new filter
                Filter.append(Bloom(len(Q_temp)+10, 0.01))
                Filter[-1].update(Q_temp)
                Q = Q + list(Q_temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
        
        message=[]
        if len(Q) >= k:
            for i in range(len(nodes)):
                # query initiator broadcast a Bloom filter of Q_temp with false positive rate 0.01
                nodes[i].nbrounds+=1
                nodes[i].received_message[nodes[i].nbrounds] = Filter[-1].size_in_bits/8+8
                start = time.time()
                next_cursor = max(len(nodes[i].darray)-np.searchsorted(nodes[i].darray[::-1], nodes[i].threshold), nodes[i].cursor)
                # collect unseen scores of items in Q (t2 <= s_ij < t1|e_i in Q)
                ind = {}
                ind[len(Filter)] = np.array([x in Filter[-1] for x in nodes[i].dindex[k:nodes[i].cursor]], dtype='bool')
                for f in range(len(Filter)):
                    ind[f] = np.array([x in Filter[f] for x in nodes[i].dindex[nodes[i].cursor:next_cursor]], dtype='bool')
                    ind[0] = ind[0] | ind[f]
                ind[len(Filter)] = np.concatenate((ind[len(Filter)], ind[0])) & ~nodes[i].sent[k:next_cursor]
                end=time.time()
                nodes[i].compute_time[nodes[i].nbrounds] = end-start
                # collect scores not smaller than t2 of items in Q_temp
                message.append([nodes[i].dindex[k:next_cursor][ind[len(Filter)]], nodes[i].darray[k:next_cursor][ind[len(Filter)]]])
                nodes[i].sent[k:next_cursor][ind[len(Filter)]] = True
                nodes[i].ID_sent[nodes[i].nbrounds] = sum(ind[len(Filter)])
                nodes[i].score_sent[nodes[i].nbrounds] = sum(ind[len(Filter)])
                nodes[i].cursor = next_cursor
            start = time.time()
            for i in range(len(nodes)):
                temp_score[message[i][0]]+=message[i][1]
                if not adapt:
                    miss[message[i][0]]-=1
            if adapt:
                miss = np.ones(n)*sum([node.threshold for node in nodes])
                for i in range(len(nodes)):
                    miss[nodes[i].dindex[nodes[i].sent]]-=nodes[i].threshold
                best = miss+temp_score+1e-5
            if not adapt:
                best = miss*nodes[0].threshold+temp_score
                b = min(b, np.partition(best, -k)[-k])
            p=np.partition(temp_score, -k)[-k]
            end=time.time()
            cct[nodes[0].nbrounds]=end-start
        else:
            for i in range(len(nodes)):
                nodes[i].cursor = n-np.searchsorted(nodes[i].darray[::-1], nodes[i].threshold)      
    
    if approx:
        start=time.time()
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        start=time.time()
        final = np.where(best>p)[0]
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
        message=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
            message.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message[i][0]]+=message[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start 
    return collected, cct

def BDBPA(nodes, n, k, b, epsilon=0, mode='c'):
    """
    Bulk Distributed Best Position Algorithm
    """
    c=b
    temp_score = np.zeros(n)
    q=0
    cct={}
    threshold = np.inf
    while q*(1+epsilon) < threshold:
        Q = set()
        threshold = 0
        message = []
        for i in range(len(nodes)):
            nodes[i].nbrounds+=1
            nodes[i].received_message[nodes[i].nbrounds] = 1
            start = time.time()
            ind = nodes[i].dindex[~nodes[i].sent][:b]
            end=time.time()
            try:
                threshold+= nodes[i].darray[~nodes[i].sent][b]
            except IndexError:
                threshold+=0
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            message.append(ind)
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        start = time.time()
        for i in range(len(nodes)):
            for x in message[i]:
                Q.add(x)
        Q = np.array(list(Q))
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
        message=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds+=1
            nodes[i].received_message[nodes[i].nbrounds] = len(Q)*8
            start = time.time()
            ind = np.where(np.isin(nodes[i].dindex, Q))[0]
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].sent[ind]=True
            message.append([nodes[i].dindex[ind], nodes[i].darray[ind]])
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        start = time.time()
        for i in range(len(nodes)):
            temp_score[message[i][0]]+=message[i][1]
        if mode=='t':
            b=b+c
        elif mode=='e':
            b=2*b
        q=np.partition(temp_score, -k)[-k]
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
    start=time.time()
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]+=end-start   
    return collected, cct

def KLEE3(nodes, n, k, c=10, a=1, adapt=False, approx=True):
    """
    KLEE implementation
    """    
    # first phase
    temp_score = np.zeros(n)
    hist_dict = {}
    hdict = {}
    average_dict = {}
    bloom_dict = {}
    cct={}
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        hist = np.histogram(nodes[i].darray, bins=100)
        average = []
        for h in range(99):
            if hist[0][h] > 0:
                average.append(sum(nodes[i].darray[(nodes[i].darray>=hist[1][h]) & (nodes[i].darray<hist[1][h+1])])/hist[0][h])
            else:
                average.append(0)
        average.append(sum(nodes[i].darray[(nodes[i].darray>=hist[1][99]) & (nodes[i].darray<=hist[1][100])])/hist[0][99])
        bloom=[]
        cur_ind = 0
        for h in range(c):
            capacity = hist[0][-(h+1)]+1
            fpf = 0.001
            B = Bloom(capacity, fpf)
            B.update(nodes[i].dindex[cur_ind:cur_ind+hist[0][-(h+1)]])
            cur_ind = cur_ind + hist[0][-(h+1)]
            bloom.append(B)
            nodes[i].extended[nodes[i].nbrounds]=B.size_in_bits/8+8
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        average_dict.update({i:average})
        hist_dict.update({i:hist})
        bloom_dict.update({i:bloom})
        if adapt:
            start = time.time()
            bins = np.percentile(nodes[i].darray, np.linspace(0, 100, 100 + 1))
            histt = np.histogram(nodes[i].darray, bins=bins)
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] += end-start
            hdict.update({i:histt})
            # save the size (in bytes) used by the histogram
            nodes[i].extended[nodes[i].nbrounds] += 100+101*8
    start = time.time()
    checkset = set()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
        checkset.update(message[i][0])
    predict_score = temp_score.copy()
    for i in range(len(nodes)):
        candidates = np.array(list(checkset-set(message[i][0])))
        low_end_ave = sum(average_dict[i][:100-c]*hist_dict[i][0][:100-c])/sum(hist_dict[i][0][:100-c])
        for j in range(c):
            checklist = np.array([ele in bloom_dict[i][j] for ele in candidates])
            predict_score[candidates[checklist]]+=average_dict[i][-(h+1)]
            candidates = candidates[~checklist]
        predict_score[candidates] += low_end_ave
    tau1=np.partition(predict_score, -k)[-k]*a
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    if adapt:
        start = time.time()
        hist = pd.DataFrame(np.zeros((100, len(nodes))))
        for i in range(len(nodes)):
            hist.iloc[0, i] = hdict[i][0][-1]
            for j in range(1, 100):
                hist.iloc[j, i] = sum(hdict[i][0][-j-1:])
        kmin = tau1/a
        # using binary process to find appropriate threshold
        temp=2*kmin
        tempup = 0
        cnumu = n
        cnumd = 0
        cnum = np.ceil((cnumu + cnumd)/2)
        while (temp > a*kmin) or (tempup < a*kmin):
            temp = 0
            tempup = 0
            cnum = np.ceil((cnumu + cnumd)/2)
            for i in range(len(nodes)):
                ind = hist.iloc[:, i][hist.iloc[:, i] <= cnum]
                if len(ind)==0:
                    ind = -1
                else:
                    ind = ind.index[-1]
                if ind == 99:
                    temp+=0
                    tempup+=hdict[i][1][-ind-2]
                else:
                    temp+=hdict[i][1][-ind-3]
                    tempup+=hdict[i][1][-ind-2]
            if temp > a*kmin:
                cnumd = cnum
            if tempup < a*kmin:
                cnumu = cnum
        for i in range(len(nodes)):
            ind = hist.iloc[:, i][hist.iloc[:, i] <= cnum]
            if len(ind)==0:
                ind = -1
            else:
                ind = ind.index[-1]
            if ind == 99:
                nodes[i].threshold = 0 + hdict[i][1][-ind-2]/tempup*(kmin*a-temp)
            else:
                nodes[i].threshold = hdict[i][1][-ind-3] + hdict[i][1][-ind-2]/tempup*(kmin*a-temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        for i in range(len(nodes)):
            nodes[i].threshold = tau1/len(nodes)
    
    miss = np.ones(n)*tau1
    message2=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8 # send the threshold to each local nodes
        start=time.time()
        nodes[i].cursor = len(nodes[i].darray)-np.searchsorted(nodes[i].darray[::-1], nodes[i].threshold)
        ind = [nodes[i].dindex[k:nodes[i].cursor], nodes[i].darray[k:nodes[i].cursor]]
        end=time.time()
        message2.append(ind)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind[0])
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message2[i][0]]+=message2[i][1]
        miss[message[i][0]]-=nodes[i].threshold
        miss[message2[i][0]]-=nodes[i].threshold
    tau2 = np.partition(temp_score, -k)[-k]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    if approx:
        start=time.time()
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        # avoid missing items in final due to very small error happened when calculating
        start=time.time()
        best = temp_score + miss +1e-5
        final = np.where(best>tau2)[0]
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    
    if not approx:
        message3=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
            message3.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message3[i][0]]+=message3[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start 
    return collected, cct

def HEGBA(nodes, n, k, delta=0.05, a=1, adapt=False, approx=True):
    """
    EGBA merge with TPUT (SHEGBA <=> approx=False)
    """
    # check condition
    if k <= 0 or k>n or delta <= 0 or delta >= 1 or a <= 0 or a > 1:
        raise Exception('Insufficient input(s)')
    
    # estimate lower bound p of s_(k)
    hdict = {}
    temp_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        if adapt:
            start = time.time()
            bins = np.percentile(nodes[i].darray, np.linspace(0, 100, 100 + 1))
            hist = np.histogram(nodes[i].darray, bins=bins)
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] += end-start
            hdict.update({i:hist})
            # save the size (in bytes) used by the histogram
            nodes[i].extended[nodes[i].nbrounds] = 100+101*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    if not adapt:
        miss = np.ones(n)*len(nodes)
        for i in range(len(nodes)):
            miss[message[i][0]]-=1
    tau1=np.partition(temp_score, -k)[-k]*a
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    if adapt:
        start = time.time()
        hist = pd.DataFrame(np.zeros((100, len(nodes))))
        for i in range(len(nodes)):
            hist.iloc[0, i] = hdict[i][0][-1]
            for j in range(1, 100):
                hist.iloc[j, i] = sum(hdict[i][0][-j-1:])
        kmin = tau1/a
        # using binary process to find appropriate threshold
        temp=2*kmin
        tempup = 0
        cnumu = n
        cnumd = 0
        cnum = np.ceil((cnumu + cnumd)/2)
        while (temp > a*kmin) or (tempup < a*kmin):
            temp = 0
            tempup = 0
            cnum = np.ceil((cnumu + cnumd)/2)
            for i in range(len(nodes)):
                ind = hist.iloc[:, i][hist.iloc[:, i] <= cnum]
                if len(ind)==0:
                    ind = -1
                else:
                    ind = ind.index[-1]
                if ind == 99:
                    temp+=0
                    tempup+=hdict[i][1][-ind-2]
                else:
                    temp+=hdict[i][1][-ind-3]
                    tempup+=hdict[i][1][-ind-2]
            if temp > a*kmin:
                cnumd = cnum
            if tempup < a*kmin:
                cnumu = cnum
        for i in range(len(nodes)):
            ind = hist.iloc[:, i][hist.iloc[:, i] <= cnum]
            if len(ind)==0:
                ind = -1
            else:
                ind = ind.index[-1]
            if ind == 99:
                nodes[i].threshold = 0 + hdict[i][1][-ind-2]/tempup*(kmin*a-temp)
            else:
                nodes[i].threshold = hdict[i][1][-ind-3] + hdict[i][1][-ind-2]/tempup*(kmin*a-temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
    else:
        for i in range(len(nodes)):
            nodes[i].threshold = tau1/len(nodes)
    
    # collect exponential candidates
    collect_threshold = -np.log(delta)/(tau1/a)
    message=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds+=1
        nodes[i].received_message[nodes[i].nbrounds] = 8
        # generate exponential random sample in range t2 <= s_ij < t1
        start = time.time()
        if not approx:
            candidates = nodes[i].dindex[(nodes[i].darray >= nodes[i].threshold*(1-a))
                                   & (np.random.exponential(1/nodes[i].darray) <= collect_threshold)]
        else:
            candidates = nodes[i].dindex[(nodes[i].darray >= nodes[i].threshold)
                                   & (np.random.exponential(1/nodes[i].darray) <= collect_threshold)]
        message.append(candidates)
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(candidates)
    start = time.time()
    Q = set()
    for i in range(len(nodes)):
        for x in message[i]:
            Q.add(x)
    Qfilter = Bloom(len(Q)+1, 0.001)
    Qfilter.update(Q)
    end = time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # first retrieval
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds+=1
        nodes[i].received_message[nodes[i].nbrounds] = Qfilter.size_in_bits/8+8
        start=time.time()
        cursor = max(len(nodes[i].darray)-np.searchsorted(nodes[i].darray[::-1], nodes[i].threshold), k)
        f = [ele in Qfilter for ele in nodes[i].dindex[k:cursor]]
        message.append([nodes[i].dindex[k:cursor][f], nodes[i].darray[k:cursor][f]])
        end = time.time()
        nodes[i].ID_sent[nodes[i].nbrounds] = sum(f)
        nodes[i].score_sent[nodes[i].nbrounds] = sum(f)
        nodes[i].cursor = max(k, cursor)
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
        if not adapt:
            miss[message[i][0]]-=1
    tau2=np.partition(temp_score, -k)[-k]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start  
    if approx:
        start=time.time()
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
        
    # second retrieval
    if not approx:
        start=time.time()
        if adapt:
            miss = np.ones(n)*tau1*(1-a)
            for i in range(len(nodes)):
                miss[nodes[i].dindex[nodes[i].sent]]-=nodes[i].threshold
            best = miss+temp_score+1e-5
        else:
            best = miss*nodes[0].threshold+temp_score+1e-5
        final = np.where(best>tau2)[0]
        end=time.time()
        cct[nodes[0].nbrounds]+=end-start
        message=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
            message.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message[i][0]]+=message[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start 
    return collected, cct

def solve_k(epsilon, delta):
    z = scipy.stats.norm.ppf(1-delta)
    kl = np.ceil(((z*(1+epsilon)+np.sqrt(z*z*(1+epsilon)**2-epsilon*z*z*(1+epsilon)))/epsilon)**2)
    ku = np.ceil((z/epsilon+np.sqrt(z/epsilon*(z+z/epsilon)))**2-1)
    
    kln = np.ceil(((1-scipy.special.gammaincinv(kl, delta)/kl)/(1-1/np.sqrt(1+epsilon)))**2*kl)
    while np.abs(kln - kl)>1:
        kl=kln
        kln = np.ceil(((1-scipy.special.gammaincinv(kl, delta)/kl)/(1-1/np.sqrt(1+epsilon)))**2*kl)
    kl = kln
    kun = np.ceil(((scipy.special.gammaincinv(ku, 1-delta)/ku-1)/(np.sqrt(1+epsilon)-1))**2*ku)
    while np.abs(kun - ku)>1:
        ku=kun
        kun = np.ceil(((scipy.special.gammaincinv(ku, 1-delta)/ku-1)/(np.sqrt(1+epsilon)-1))**2*ku)
    ku = kun
    return max([kl, ku])