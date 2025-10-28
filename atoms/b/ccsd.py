import numpy as np
from pyscf import gto, scf, cc

norb_frozen = 1
s = 1
atomstring = f'''
B   0.000000   0.000000   0.000000
'''

mol = gto.M(atom = atomstring, verbose=4, basis='ccpvdz', spin=s)
mf = scf.RHF(mol)
# mf.chkfile = 'mf.chk'
mf.max_cycle = 500
mf.level_shift = 0.4
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mf.stability()

mycc = cc.CCSD(mf)
mycc.frozen = norb_frozen
mycc.run()
et = mycc.ccsd_t()
print(mycc.e_tot + et)
