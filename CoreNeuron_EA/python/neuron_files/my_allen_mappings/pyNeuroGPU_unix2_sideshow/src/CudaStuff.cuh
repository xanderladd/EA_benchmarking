#ifndef __CUDASTUFF__
#define __CUDASTUFF__

#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include "Util.h"
#include <cusparse.h>
extern __shared__ char smem[];

void stEfork2Main(Stim stim, Sim sim, MYFTYPE* ParamsM, MYFTYPE* cm_input, MYFTYPE* InitStatesM, HMat& InMat, MYFTYPE* V,MYDTYPE CompDepth,MYDTYPE CompFDepth,int NSets,int *p2pCapableGPUs,int np2p, int stim_ind, int globalRank);
void stEfork2TimeLoopGPU(Stim stim, Sim sim, MYFTYPE* ParamsM, MYFTYPE* InitStatesM, HMat& InMat, MYFTYPE* V,MYDTYPE CompDepth,MYDTYPE CompFDepth,MYDTYPE NSets);
void initFrameWork(Stim stim, Sim sim, MYFTYPE* ParamsM, MYFTYPE* InitStatesM, HMat& InMat, MYDTYPE CompDepth, MYDTYPE CompFDepth, MYDTYPE NSets, HMat& Mat_d);
void ReadParamsMatX(const char* FN, MYFTYPE* ParamsM,MYDTYPE NParams,MYDTYPE Nx);
void ReadParamsMat(const char* FN, MYFTYPE** ParamsM,MYDTYPE NParams,MYDTYPE Nx);
MYFTYPE* ReadAllParams(const char* FN, MYDTYPE NParams,MYFTYPE* cm_input, MYDTYPE Nx,MYDTYPE &NSets);
void FreeParallelNeuronData(HMat &InMat);

#endif