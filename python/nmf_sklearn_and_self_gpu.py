#!/usr/bin/env python3

import pandas as pd
import numpy as np

import argparse
import os.path
import sys
import time

import sklearn.decomposition as skdc

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

#
# Function for down-sampling the input data
# the down-sampling is done by averaging input data within the time bin, 
# while the time bins are not overlapped.
#

def down_sample(data_in, bin=1):
    data_tmp = np.empty((0, data_in.shape[1]))
    for i in range(0,data_in.shape[0],bin):
        data_tmp = np.append(data_tmp, [np.mean(data_in[i:i+bin,:], axis=0)], axis=0)

    return data_tmp

#
# This function attempts to research for solution (based on SKlearn)
#

@ignore_warnings(category=ConvergenceWarning)
def solve_NMF_sk(data_drift, k, Na=10):

    matrix_B = np.zeros((data_drift.shape[0], k)) 
    matrix_C = np.zeros((k, data_drift.shape[1])) 

    best = -1

    for i in range(Na):

        if   i == 0:
            init_method = "nndsvd"
        elif i == 1:          
            init_method = "nndsvda"
        elif i == 2:          
            init_method = "nndsvdar"
        else:                 
            init_method = "random"

        data_drift_work = 0.5 * (data_drift + np.abs(data_drift))
        model = skdc.NMF(n_components=k, init=init_method, random_state=i, max_iter=100, solver="cd")
        matrix_B_i = model.fit_transform(data_drift_work)
        matrix_C_i = model.components_

        data_drift_work = 0.5 * (data_drift + np.abs(data_drift))
        model = skdc.NMF(n_components=k, init='custom', max_iter=10000, solver="cd", tol=1e-10)
        matrix_B_i = model.fit_transform(data_drift_work, W=matrix_B_i, H=matrix_C_i)
        matrix_C_i = model.components_

        diff = np.sum(np.square(data_drift - np.matmul(matrix_B_i, matrix_C_i)))

        if diff < best or best < 0:
            best = diff
            matrix_B = matrix_B_i
            matrix_C = matrix_C_i

    return matrix_B, matrix_C, best

#
# This function attempts to research for solution (self-implemented GPU)
#

def solve_NMF_gpu(data_drift_in, k, fp_type, Na=10, Nt=10000):
    data_drift_0 = cupy.array((data_drift_in + np.abs(data_drift_in)) * 0.5, dtype=fp_type)
        
    data_mean_root = cupy.sqrt(cupy.mean(data_drift_0)/k)

    data_mean = cupy.mean(data_drift_0)

    best = -1

    for i in range(Na):

        if (i//2) == 1 :
            u,s,v = cupy.linalg.svd(data_drift_0)

            for idx in range(s.shape[0]):
                u[:,idx] *= cupy.sqrt(s[idx])
                v[idx,:] *= cupy.sqrt(s[idx])

            matrix_B_i = (0.5 * (u + cupy.abs(u)))[:,0:k] + 0.1 * data_mean_root
            matrix_C_i = (0.5 * (v + cupy.abs(v)))[0:k,:] + 0.1 * data_mean_root
        else:
            cupy.random.seed(k+(i//2))
            matrix_B_i = (cupy.random.rand(data_drift_0.shape[0], k) + data_mean_root).astype(fp_type)
            matrix_C_i = (cupy.random.rand(k, data_drift_0.shape[1]) + data_mean_root).astype(fp_type)

        for t0 in range(Nt):
            
            matrix_B_i += 1e-10
            matrix_C_i += 1e-10

            data_drift = cupy.copy(data_drift_0) 

            if t0 < (Nt - 500.0):
                tmp = 1.0 - cupy.exp(-(t0 / 200.0))
                alpha_0 = tmp * cupy.random.rand(matrix_B_i.shape[0], matrix_B_i.shape[1])
                alpha_1 = tmp * cupy.random.rand(matrix_C_i.shape[0], matrix_C_i.shape[1])
            else:
                alpha_0 = 1.0 - cupy.exp(-(t0 / 200.0))
                alpha_1 = 1.0 - cupy.exp(-(t0 / 200.0))
                data_drift += cupy.exp(-(t0 / 200.0)) * data_mean * \
                         (cupy.random.rand(data_drift_0.shape[0], data_drift_0.shape[1]) + 0.5)

            
            if (t0%2) == 0:
                d_mat_B_i_1 = cupy.matmul(data_drift, matrix_C_i.T)
                d_mat_B_i_2 = cupy.matmul(cupy.matmul(matrix_B_i, matrix_C_i), matrix_C_i.T) + 1e-10
                d_mat_B_i = cupy.power(cupy.divide(d_mat_B_i_1, d_mat_B_i_2), alpha_0).astype(fp_type)
                matrix_B_i = matrix_B_i * d_mat_B_i

            d_mat_C_i_1 = cupy.matmul(matrix_B_i.T, data_drift)
            d_mat_C_i_2 = cupy.matmul(matrix_B_i.T, cupy.matmul(matrix_B_i, matrix_C_i)) + 1e-10
            d_mat_C_i = cupy.power(cupy.divide(d_mat_C_i_1, d_mat_C_i_2), alpha_1).astype(fp_type)
            matrix_C_i = matrix_C_i * d_mat_C_i
            
            if (t0%2) == 1:
                d_mat_B_i_1 = cupy.matmul(data_drift, matrix_C_i.T)
                d_mat_B_i_2 = cupy.matmul(cupy.matmul(matrix_B_i, matrix_C_i), matrix_C_i.T) + 1e-10
                d_mat_B_i = cupy.power(cupy.divide(d_mat_B_i_1, d_mat_B_i_2), alpha_0).astype(fp_type)
                matrix_B_i = matrix_B_i * d_mat_B_i
 
            diff = cupy.sum(cupy.square(data_drift - cupy.matmul(matrix_B_i, matrix_C_i)))

        if diff < best or best < 0:
            best = diff
            matrix_B = matrix_B_i
            matrix_C = matrix_C_i

    return cupy.asnumpy(matrix_B), cupy.asnumpy(matrix_C), best

########################################################
########################################################
########################################################

#
# Main Program
#

# Getting parameters from program arguments

parser = argparse.ArgumentParser(description="")

parser.add_argument("-v", help="Verbose", action="store_true")

p_input = parser.add_argument_group("Input File Control")

p_input.add_argument("-i",   metavar="string", type=str,  help="Data Set", required=True)
p_input.add_argument("-mat", help="Switch for mat input file", action="store_true")
p_input.add_argument("-fa",  metavar="int", type=int, help="Starting timeframe")
p_input.add_argument("-fb",  metavar="int", type=int, help="Ending timeframe")
p_input.add_argument("-Nd",  metavar="int", type=int, help="Number of bins to down-sample", default=4)

p_order = parser.add_argument_group("Order Control for NMF")

p_order.add_argument("-Ns",  metavar="int", type=int, help="Starting order", default=1)
p_order.add_argument("-Ne",  metavar="int", type=int, help="Ending order", default=300)
p_order.add_argument("-Na",  metavar="int", type=int, help="Number of attempts for each order", default=10)
p_order.add_argument("-smart_search", help="Smart sarech for a narrower range of orders for the best-fit (-Ns and -Ne will be ignored)", action="store_true")
p_order.add_argument("-S", help="Same as -smart_search", action="store_true")

p_output = parser.add_argument_group("Output File Control")

p_output.add_argument("-ob",  metavar="string", type=str,  help="Output for Patterns")
p_output.add_argument("-oc",  metavar="string", type=str,  help="Output for Occurrence")
p_output.add_argument("-oe",  metavar="string", type=str,  help="Output for Costs and AICc")
p_output.add_argument("-om",  metavar="string", type=str,  help="Output for Reconstructed Matrix")
p_output.add_argument("-y", help="Ignore the existence of output file", action="store_true")

p_numerical = parser.add_argument_group("Control of Numerical Processing")

p_numerical.add_argument("-no_daz", help="Does not ignore de-normal numbers", action="store_true")
p_numerical.add_argument("-use_gpu", help="Using self-implemented GPU method with FP32 (default no)", action="store_true")
p_numerical.add_argument("-use_gpu_fp64", help="Using self-implemented GPU method with FP64, VERY SLOW on some devices (default no)", action="store_true")

arg = parser.parse_args()

# Check if output files exist

if not arg.y:

    if arg.ob is not None and os.path.exists(arg.ob):
        print("Error: output file ", arg.ob ," exists.", sep="", file=sys.stderr)
        exit(1)

    if arg.oc is not None and os.path.exists(arg.oc):
        print("Error: output file ", arg.oc ," exists.", sep="",  file=sys.stderr)
        exit(1)

    if arg.oe is not None and os.path.exists(arg.oe):
        print("Error: output file ", arg.oe ," exists.", sep="",  file=sys.stderr)
        exit(1)

    if arg.om is not None and os.path.exists(arg.om):
        print("Error: output file ", arg.om ," exists.", sep="",  file=sys.stderr)
        exit(1)

# If use_gpu is on, try to import cupy

if arg.use_gpu or arg.use_gpu_fp64:
    try:
        import cupy
    except:
        print("module cupy is not installed", flush=True, file=sys.stderr)
        exit(1)
    
# Timer Start

time_start = time.time()

# Set daz

if not arg.no_daz:
    try:
        import daz
        daz.set_daz()
    except ImportError as e:
        pass

# Getting data from input file

data_raw = np.zeros(1)

if arg.mat :

# Getting data from mat file

    import h5py
    datafile = h5py.File(arg.i, 'r')
    data_raw = np.array(list(datafile.items())[0][1])
    datafile.close()

else:

# Getting data from the csv file

    data_raw = (pd.read_csv(arg.i, sep='[,;\t]', dtype=np.float64, header=None, engine='python')).values

# Checking how the data is orientated
# Here, I assume the longer edge is the time axis

if data_raw.shape[0] < data_raw.shape[1]:
    data_raw = data_raw.T

# Selecting rows of the table between arg.fa and arg.fb
if arg.fb is not None:
    data_raw = data_raw[:int(arg.fb),:]
if arg.fa is not None:
    data_raw = data_raw[int(arg.fa-1):,:]

# Checking whether the first column looks like time labels
# Here the checking is done by correlation of the column and a sequence of natural numbers

if (np.corrcoef(np.array(range(data_raw.shape[0])).T, data_raw[:,0])).min() > 0.95:
    data_raw = np.delete(data_raw, 0, 1)

# Down-sampling the data

data_down = down_sample(data_raw, arg.Nd)

data_processed = data_down 

# Variables storing best result

event_count = 0

AICc_best = 0

best_matrix_B = np.zeros((1,1))

best_matrix_C = np.zeros((1,1))

# Transpose the matrix to make their orientations the same as that in the derivation

if arg.use_gpu:
    data_processed = cupy.array(data_processed.T, cupy.float32)
elif arg.use_gpu_fp64:
    data_processed = cupy.array(data_processed.T, cupy.float64)
else:
    data_processed = data_processed.T

# Summary Message of the Process

Summary = \
"#" + "\n" + \
"# *** Summary for the NMF analysis ***" + "\n" + \
"# Range of order:\t " + str(arg.Ns) + " - " + str(arg.Ne) + "\n" + \
"# Number of Attempts:\t" + str(arg.Na) + "\n" + \
"#" + "\n" + \
"# Input Data File:\t" + str(arg.i) + "\n" + \
"#" + "\n" + \
"# Output for Cost and AIC:\t\t" + str(arg.oe) + "\n" + \
"# Output for Basis:       \t\t" + str(arg.ob) + "\n" + \
"# Output for Occurrence:   \t\t" + str(arg.oc) + "\n" + \
"# Output for Reconstructed Mtx:   \t" + str(arg.om) + "\n" + \
"#"   

# Print the header of the output files

if arg.oe is not None:
    with open(arg.oe, 'w') as fout:
        print("# Minimization Cost of NMF and Corresponding AICc", flush=True, file=fout)
        print(Summary, flush=True, file=fout)

if arg.v:
    print("# Minimization Cost of NMF and Corresponding AICc", flush=True, file=sys.stdout)
    print(Summary, flush=True, file=sys.stdout)

########################################################
# Searching for a range of order with only one iteration
########################################################

best_order = -1;

orders_record = list()
RSS_record = list()
AICs_record = list()

if arg.S or arg.smart_search:

    for order in range(1,int(data_processed.shape[0]*0.8),3):

        # Solve the NMF

        if   arg.use_gpu:
            matrix_B, matrix_C, RSS = solve_NMF_gpu(data_processed, order, fp_type=cupy.float32, Na=1)
        elif arg.use_gpu_fp64:
            matrix_B, matrix_C, RSS = solve_NMF_gpu(data_processed, order, fp_type=cupy.float64, Na=1)
        else:
            matrix_B, matrix_C, RSS = solve_NMF_sk(data_processed, order, Na=1)

        # Number of parameters

        k = matrix_B.shape[0] * matrix_B.shape[1] + matrix_C.shape[0] * matrix_C.shape[1]

        # Number of observed data

        n = data_processed.shape[0] * data_processed.shape[1]

        # if the number of parameters is more than the number of 
        # observations, the calculation stops

        if k >= n:
            break

        # Calculation the Akaike information critirtion (AIC)

        AICc = n * np.log(RSS / n) + 2.0 * k + (2.0 * k * k + 2.0 * k) / (n - k - 1)

        if best_order < 0 or AICc < AICc_best:
            AICc_best  = AICc
            best_order = order

        orders_record.append(order)
        RSS_record.append(RSS)
        AICs_record.append(AICc)

        if order > np.max([best_order + 10, best_order * 1.3]):
            break

    # Making the preliminary saerch effective

    d_order = int(np.max([10, best_order * 0.2]))

    arg.Ns = best_order - d_order
    arg.Ne = best_order + d_order

    if arg.Ns < 1: 
        arg.Ns = 1

# Resetting Variables storing best result

event_count = 0

AICc_best = 0

if arg.oe is not None:
    with open(arg.oe, 'a') as fout:
        print("# Time used for initialization:\t", time.time() - time_start, flush=True, file=fout)
        print("#", flush=True, file=fout)
        print("#Order\tE\tAICc\tTime_Used", flush=True, file=fout)

        if len(orders_record):
            for idx in range(len(orders_record)):
                if orders_record[idx] < arg.Ns:
                    print(orders_record[idx], RSS_record[idx], AICs_record[idx], 0 , sep="\t", end="\n", flush=True, file=fout)

if arg.v:
    print("# Time used for initialization:\t", time.time() - time_start, flush=True, file=sys.stdout)
    print("#", flush=True, file=sys.stdout)
    print("#Order\tE\tAICc\tTime_Used", flush=True, file=sys.stdout)

    if len(orders_record):
        for idx in range(len(orders_record)):
            if orders_record[idx] < arg.Ns:
                print(orders_record[idx], RSS_record[idx], AICs_record[idx], 0 , sep="\t", end="\n", flush=True, file=sys.stdout)

AICc_old = None

#####################
# Solve the equation
#####################

for order in range(arg.Ns, arg.Ne+1):

    time_start = time.time()

    # Solve the NMF

    if  arg.use_gpu:
        matrix_B, matrix_C, RSS = solve_NMF_gpu(data_processed, order, fp_type=cupy.float32, Na=arg.Na)
    elif arg.use_gpu_fp64:
        matrix_B, matrix_C, RSS = solve_NMF_gpu(data_processed, order, fp_type=cupy.float64, Na=arg.Na)
    else:    
        matrix_B, matrix_C, RSS = solve_NMF_sk(data_processed, order, Na=arg.Na)

    # Number of parameters

    k = matrix_B.shape[0] * matrix_B.shape[1] + matrix_C.shape[0] * matrix_C.shape[1]

    # Number of observed data

    n = data_processed.shape[0] * data_processed.shape[1]

    # if the number of parameters is more than the number of 
    # observations, the calculation stops

    if k >= n:
        break


    # Calculation the Akaike information critirtion (AIC)

    AICc = n * np.log(RSS / n) + 2.0 * k + (2.0 * k * k + 2.0 * k) / (n - k - 1)

    if AICc_old is not None and AICc < AICc_best:
        if np.abs((AICc - AICc_old) / (AICc - AICc_best)) > 10.0:
            break

    AICc_old = AICc

    # If the result of the current order is better (lower AIC), then they are stored in corresponding variables

    if event_count < 1 or AICc < AICc_best :
        best_matrix_B = matrix_B
        best_matrix_C = matrix_C
        AICc_best = AICc
        event_count += 1

    time_used = time.time() - time_start
 
    # if option -oe is on, output the cost and AIC to the file arg.oe

    if arg.oe is not None:
        with open(arg.oe, 'a') as fout:
            print(order, RSS, AICc, time_used , sep="\t", end="\n", flush=True, file=fout)

    # if option -v is on, output to the command line
            
    if arg.v:
        print(order, RSS, AICc, time_used, sep="\t", flush=True)

if arg.oe is not None:
    with open(arg.oe, 'a') as fout:
        if len(orders_record):
            for idx in range(len(orders_record)):
                if orders_record[idx] > arg.Ne:
                    print(orders_record[idx], RSS_record[idx], AICs_record[idx], 0 , sep="\t", end="\n", flush=True, file=fout)

if arg.v:
    if len(orders_record):
        for idx in range(len(orders_record)):
            if orders_record[idx] > arg.Ne:
                print(orders_record[idx], RSS_record[idx], AICs_record[idx], 0 , sep="\t", end="\n", flush=True, file=sys.stdout)

# Output the basis (or feature) to the file arg.ob

if arg.ob is not None:
    with open(arg.ob, 'w') as fout:
        print("# Optimal basis from NMF", flush=True, file=fout)
        print(Summary, flush=True, file=fout)
        print("#Unit\tPattern1\tPattern2\t...", flush=True, file=fout)

if arg.ob is not None:
    with open(arg.ob, 'a') as fout:
        for i in range(best_matrix_B.shape[0]):
            print(i, end="\t", flush=True, file=fout)
            for j in range(best_matrix_B.shape[1]):
                print(best_matrix_B[i,j], end="\t", flush=True, file=fout)
            print(end="\n", flush=True, file=fout)

# Output the occurrence of the basis to the file arg.oc

if arg.oc is not None:
    with open(arg.oc, 'w') as fout:
        print("# Corresponding Occurrence of basis from NMF", flush=True, file=fout)
        print(Summary, flush=True, file=fout)
        print("#Time\tOccur1\tOccur2\t...", flush=True, file=fout)

if arg.oc is not None:
    with open(arg.oc, 'a') as fout:
        for i in range(best_matrix_C.shape[1]):
            print(i, end="\t", flush=True, file=fout)
            for j in range(best_matrix_C.shape[0]):
                print(best_matrix_C[j,i], end="\t", flush=True, file=fout)
            print(end="\n", flush=True, file=fout)

# Output the reconstructed matrix to the file arg.or

if arg.om is not None:
    with open(arg.om, 'w') as fout:
        print(Summary, flush=True, file=fout)
        print("#Time\tOccur1\tOccur2\t...", flush=True, file=fout)

if arg.om is not None:

    re_mtx = np.matmul(best_matrix_B, best_matrix_C).transpose()

    with open(arg.om, 'a') as fout:
        for i in range(re_mtx.shape[0]):
            print(i, end="\t", flush=True, file=fout)
            for j in range(re_mtx.shape[1]):
                print(re_mtx[i,j], end="\t", flush=True, file=fout)
            print(end="\n", flush=True, file=fout)

sys.exit(0)
