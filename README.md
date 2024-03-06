# iterative-top-k

# Version:

- Python 3.11.5
- numpy 1.24.3
- pandas 2.0.3
- multiprocess 0.70.14
- rbloom 1.5.0 (for fast Bloom filter on python)

# How to use:

- Synthesis data: First, set the value for parameters: alpha - skewness of global distribution, n - number of distinct items, m - number of nodes, dis - how scores of items are partitioned among nodes. Then, run functions in utils.common to generate data. Last, call functions from utils.method to run directly. The example is shown in main.py
