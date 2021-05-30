import os
import time
import subprocess

while True:
    os.system("bjobs > test.txt")
    launch=True
    with open("test.txt",'r') as f:
        for line in f.readlines():
            if 'debug' in line:
                launch = False
    if launch:
        print('launching.....')
        pipe = subprocess.Popen(['bsub', 'debug_ga.lsf'])
         
#         pip = subprocess.Popen(['bsub', '-nnodes 10' ,'-W 30' ,'-P nro106', '-J gen_alg', \
#                                 '-q debug', '-o logs/', '-M 619326', """ "sh debug_ga.sh" """])
        time.sleep(60)
    else:
        time.sleep(60)
