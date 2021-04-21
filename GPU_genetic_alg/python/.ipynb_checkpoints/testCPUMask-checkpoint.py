# Python program to explain os.sched_setaffinity() method   
  
# importing os module  
import os 
from mpi4py import MPI
import os
import platform

print("="*40, "System Information", "="*40)
uname = platform.uname()
print(f"System: {uname.system}")
print(f"Node Name: {uname.node}")
print(f"Release: {uname.release}")
print(f"Version: {uname.version}")
print(f"Machine: {uname.machine}")
print(f"Processor: {uname.processor}")

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()
# in the system 
# using os.cpu_count() method 
print("Number of CPUs:", os.cpu_count()) 
  
# Get the set of CPUs 
# on which the calling process 
# is eligible to run. using 
# os.sched_getaffinity() method 
# 0 as PID represnts the 
# calling process 
pid = 0
affinity = os.sched_getaffinity(pid) 
  
# Print the result 
print("Process {} is eligibl to run on:".format(global_rank), affinity) 
  