# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import multiprocessing
from datetime import datetime
import time
from machine import local_nodes as node
from utils import common as cm
from utils import method as mt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import gc

def preprocess(i):
    i.data = i.data.sort_values(ascending=False)
    i.data = i.data[i.data>0]
    i.dup_data()
    return i
        
if __name__ == "__main__":
    result = []
    for alpha in [0.5]:
        for n in [2**20]:
            for m in [2**10]:
                for dis in ['zipf']:
                    for j in range(10):
                        print('loop' + str(j))
                        source = cm.generate_zipf_array(alpha, n, n, m*n*1000, permutate=(False))
                        
                        num_workers = 16
                        chunk = int(n/num_workers)
                        with multiprocessing.Pool(processes=num_workers) as pool:
                            data = pool.starmap(
                                cm.create_data, 
                                [(source[i*chunk:(i+1)*chunk], m, dis, 0.7,) 
                                 for i in range(num_workers)])
                        data = np.concatenate(data)
                        data = pd.DataFrame(data)
                        nodes=[]
                        for i in range(m):
                            nodes.append(node.node(data.iloc[:, i], data.index))
                        del data
                        gc.collect()
                        
                        gs = pd.Series([0]*n)
                        nodes[0].fullindex = gs.index
                        for i in range(m):
                            gs+=nodes[i].data
                        gs.sort_values(ascending=False, inplace=True)
                        
                        with multiprocessing.Pool(processes=num_workers) as pool:
                            nodes = pool.map(preprocess, nodes)
                        
                        def ktest1(k):
                            output, cct = mt.TPUT(nodes, n, k, a=0.5, adapt=True)
                            gc.collect()
                            return mt.evaluate('TPUT '+str(k), nodes, k, output, cct, gs, epsilon=0)
                        
                        def ktest2(k):
                            output, cct = mt.ITT(nodes, n, k)
                            gc.collect()
                            return mt.evaluate('ITT '+str(k), nodes, k, output, cct, gs, epsilon=0)
                        
                        def ktest3(k):
                            output, cct = mt.BDBPA(nodes, n, k, k, mode='e')
                            gc.collect()
                            return mt.evaluate('BDBPA '+str(k), nodes, k, output, cct, gs, epsilon=0)
                        
                        def ktest4(k):
                            output, cct = mt.FRUT(nodes, n, k, a=0.5, approx=False, adapt=True)
                            gc.collect()
                            return mt.evaluate('4RUT '+str(k), nodes, k, output, cct, gs, epsilon=0)
                        
                        def ktest5(k):
                            output, cct = mt.TPUT(nodes, n, k, a=1, approx=True, adapt=True)
                            gc.collect()
                            return mt.evaluate('ATPUT '+str(k), nodes, k, output, cct, gs, epsilon=1)
                        
                        def ktest6(k):
                            output, cct = mt.IES(nodes, n, k, delta=0.05, epsilon=1, approx=True)
                            gc.collect()
                            return mt.evaluate('IES '+str(k), nodes, k, output, cct, gs, epsilon=1)
                        
                        def ktest7(k):
                            output, cct = mt.samplingTopk(nodes, n, k, delta=0.05, epsilon=1)
                            gc.collect()
                            return mt.evaluate('BSTk '+str(k), nodes, k, output, cct, gs, epsilon=1)
                        
                        def ktest8(k):
                            output, cct = mt.KLEE3(nodes, n, k, a=1, approx=True, adapt=True)
                            gc.collect()
                            return mt.evaluate('KLEE '+str(k), nodes, k, output, cct, gs, epsilon=1)
                        
                        def ktest9(k):
                            output, cct = mt.ITT(nodes, n, k, a=1, approx=True)
                            gc.collect()
                            return mt.evaluate('ITT '+str(k), nodes, k, output, cct, gs, epsilon=0)
                        
                        K = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                        with multiprocessing.Pool(processes=num_workers) as pool:
                            result1 = pool.starmap(ktest9, [(k,) for k in K])
                            # result2 = pool.starmap(ktest2, [(k,) for k in K])
                            # result3 = pool.starmap(ktest3, [(k,) for k in K])
                            # result4 = pool.starmap(ktest4, [(k,) for k in K])
                            # result5 = pool.starmap(ktest5, [(k,) for k in K])
                            # result6 = pool.starmap(ktest6, [(k,) for k in K])
                            # result7 = pool.starmap(ktest7, [(k,) for k in K])
                            # result8 = pool.starmap(ktest8, [(k,) for k in K])
                        
                        result+=result1
                        gc.collect()
                        
                        k=50                      
                        def dtest6(d):
                            output, cct = mt.IES(nodes, n, k, delta=d, epsilon=1, approx=True)
                            gc.collect()
                            return mt.evaluate('IESd '+str(d), nodes, k, output, cct, gs, epsilon=1)
                        
                        def dtest7(d):
                            output, cct = mt.samplingTopk(nodes, n, k, delta=d, epsilon=1)
                            gc.collect()
                            return mt.evaluate('BSTkd '+str(d), nodes, k, output, cct, gs, epsilon=1)
                        
                        D = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]                
                        # with multiprocessing.Pool(processes=num_workers) as pool:
                        #     resultd6 = pool.starmap(dtest6, [(d,) for d in D])
                        #     resultd7 = pool.starmap(dtest7, [(d,) for d in D])
                        # result=result+resultd6+resultd7                            
                        
                        def etest5(e):
                            output, cct = mt.TPUT(nodes, n, k, a=e, approx=True, adapt=True)
                            gc.collect()
                            return mt.evaluate('ATPUTe '+str(e), nodes, k, output, cct, gs, epsilon=e)
                        
                        def etest6(e):
                            output, cct = mt.IES(nodes, n, k, delta=0.05, epsilon=e, approx=True)
                            gc.collect()
                            return mt.evaluate('IESe '+str(e), nodes, k, output, cct, gs, epsilon=e)
                        
                        def etest7(e):
                            output, cct = mt.samplingTopk(nodes, n, k, delta=0.05, epsilon=e)
                            gc.collect()
                            return mt.evaluate('BSTke '+str(e), nodes, k, output, cct, gs, epsilon=e)
                        
                        def etest8(e):
                            output, cct = mt.KLEE3(nodes, n, k, a=e, approx=True, adapt=True)
                            gc.collect()
                            return mt.evaluate('KLEEe '+str(e), nodes, k, output, cct, gs, epsilon=e)
                        
                        def etest9(e):
                            output, cct = mt.ITT(nodes, n, k, a=1, approx=True)
                            gc.collect()
                            return mt.evaluate('ITT '+str(e), nodes, k, output, cct, gs, epsilon=1)
                        
                            
                        E = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        with multiprocessing.Pool(processes=num_workers) as pool:
                            resulte5 = pool.starmap(etest9, [(e,) for e in E])
                            # resulte6 = pool.starmap(etest6, [(e,) for e in E])
                            # resulte7 = pool.starmap(etest7, [(e,) for e in E])
                            # resulte8 = pool.starmap(etest8, [(e,) for e in E])
                        result=result+resulte5
                        
                        
    

    

