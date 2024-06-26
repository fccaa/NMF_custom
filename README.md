This is a self-implementation of non-negative matrix factorization (NMF). This is specialized to perform pattern separation for the calcium signal detected by the HOTARU online sorting system (Takekawa, T. et al, 2017).

Please ues the python version in the folder named "python"

Usage can be found using the following command:

``python3 nmf_sklearn_and_self_gpu.py -h``

One may use the files in the "demo" folder for practice. By runing the following command, we can obtain the separated patterns and the corresponding occurrence.

``python3 ./nmf_sklearn_and_self_gpu.py -S -smart_search -i finalv135_01_W1.txt -Ns 1 -Ne 300 -Na 100 -oe finalv135_01_W1_energy.txt -ob finalv135_01_W1_basis.txt -oc finalv135_01_W1_coeff.txt``

Here the columns in finalv135_01_W1_basis.txt represent the neuronal activation patterns, the columns in finalv135_01_W1_coeff.txt represents the corresponding occurrence, and the rows in finalv135_01_W1_energy.txt contain the information of cost function and AIC for each NMF order.
