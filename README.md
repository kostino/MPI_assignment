# Local results

|               | Sequential |    MPI   |
|:-------------:|:----------:|:--------:|
|  n=1000 d=10  |  0.016793  | 0.060921 |
|  n=1000 d=100 |  0.022823  | 0.358121 |
|  n=10000 d=10 |  3.548600  | 6.041010 |
| n=10000 d=100 |  3.303697  | 36.70688 |

# Cluster result
* for seq : 1 node x 8 tasks
* for mpi : 4 node x 8 tasks

|               | Sequential | MPI Synchronous | MPI Asynchronous |
|:-------------:|:----------:|:---------------:|:----------------:|
|  n=1000 d=10  |  0.100000  |     0.084138    |     0.072691     |
|  n=1000 d=50  |  0.160000  |     0.104513    |     0.102335     |
|  n=1000 d=100 |  0.240000  |     0.133326    |     0.128446     |
|  n=2500 d=100 |  1.560000  |     0.48417     |     0.473638     |
|  n=5000 d=100 |  6.250000  |     1.744948    |     1.727909     |
|  n=7500 d=100 |  16.220000 |     3.995789    |     3.973511     |
|  n=10000 d=10 |  9.810000  |     2.851463    |     2.845754     |
|  n=10000 d=50 |  15.910000 |     4.149736    |     4.137334     |
| n=10000 d=100 |  33.160000 |     6.842336    |     6.801902     |

