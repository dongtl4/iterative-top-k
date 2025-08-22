# iterative-top-k

# Version:

- Python 3.11.5
- numpy 1.24.3
- pandas 2.0.3
- multiprocess 0.70.14
- rbloom 1.5.0 (for fast Bloom filter on Python)

# How to use:

- Synthesis data: First, set the value for parameters: alpha - skewness of global distribution, n - number of distinct items, m - number of nodes, dis - how scores of items are partitioned among nodes. Then, run functions in utils/common.py to generate data. Last, call functions from utils/method.py to run top-k queries. An example is shown in main.py.

- HIGGS data: download the dataset from https://archive.ics.uci.edu/dataset/280/higgs. Then, mploying functions in utils/common.py to increase the dimension of the dataset. Finally, directly call functions from utils/method.py to run top-k queries. An example is shown in mainhiggs.py.
