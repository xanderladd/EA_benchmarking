from neuron import h
from neuron import coreneuron
from neuron.units import ms, mV
import time
import numpy as np


    
class L5_TTPC1_CELL:
    """
    cell instantion for coreneuron. creates an instance of  a cADpyr232_L5_TTPC1_0fb1ca4724 cell.
    sets up a dummy stimulus at 2 picoAmps
    """
    def __init__(self, gid, pc, run):
#         print(f'BEGIN: the mpi_id is {pc.id()} and gid is {gid}')
        start = time.time()
        self.hoc_cell = h.cADpyr232_L5_TTPC1_0fb1ca4724()
        end = time.time()
#         print(f'took {end - start}') 
        self.gid = gid
        self.pc = pc
        if run == 0:
            self.pc.set_gid2node(gid, pc.id())
        # curr_p = param_list  # TODO ----> #[gid]*gid
        # self.update_params(curr_p)
        self.ic = h.IClamp(self.soma[0](.5))
        self.ic.amp = 0.5
        self.ic.dur = 1e9
        v = h.Vector().from_python(np.repeat(2,10000))
        v.play(self.ic, self.ic._ref_amp, True)
        self.rd = {k: h.Vector().record(v, sec=self.soma[0]) for k,v in zip(['t', 'v', 'stim_i', 'amp'],
                                                    [h._ref_t, self.soma[0](.5)._ref_v, self.ic._ref_i, self.ic._ref_amp])}
        
    
    
    def __getattr__(self, name):
        # we don't have it, see if the hoc_cell has it?
        return getattr(self.hoc_cell, name)
    
    def update_params(self,p):
        """
        use vector of p params to change parameters in core neuron
        """
        for curr_sec in self.hoc_cell.all:
            curr_sec.g_pas = p[0]
            curr_sec.e_pas = p[1]
        for curr_sec in self.hoc_cell.axonal:
            curr_sec.gNaTa_tbar_NaTa_t = p[2]
            curr_sec.gK_Tstbar_K_Tst = p[3]
            curr_sec.gNap_Et2bar_Nap_Et2 = p[4]
            curr_sec.gK_Pstbar_K_Pst = p[5]
            curr_sec.gSKv3_1bar_SKv3_1 = p[6]
            curr_sec.gCa_LVAstbar_Ca_LVAst = p[7]
            curr_sec.gCa_HVAbar_Ca_HVA = p[8]
        for curr_sec in self.hoc_cell.somatic:
            curr_sec.gSKv3_1bar_SKv3_1 = p[9]
            curr_sec.gCa_HVAbar_Ca_HVA = p[10]
            curr_sec.gNaTs2_tbar_NaTs2_t = p[11]
            curr_sec.gCa_LVAstbar_Ca_LVAst = p[12]
        for curr_sec in self.hoc_cell.basal:
            curr_sec.gIhbar_Ih = p[13]
        for curr_sec in self.hoc_cell.apical:
            curr_sec.gIhbar_Ih = p[13]
           