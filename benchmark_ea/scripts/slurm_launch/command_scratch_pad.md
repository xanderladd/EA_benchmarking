# Salloc Cmds
salloc -C gpu -N 1 -t 220  --ntasks-per-node 8  -c 10 --gres=gpu:8  -A m3513


# Test Launching
sh directed_shell_launcher.sh /global/cscratch1/sd/zladd/benchmark_ea/benchmark_ea/data/NeuroGPU_EA/ neurogpu_batch.sh (works)


sh directed_shell_launcher.sh /global/cscratch1/sd/zladd/benchmark_ea/benchmark_ea/data/Neuron_EA/python neuron_batch.sh 



sh directed_shell_launcher.sh /global/cscratch1/sd/zladd/benchmark_ea/benchmark_ea/data/CoreNeuron_EA/python coreneuron_batch.sh 


