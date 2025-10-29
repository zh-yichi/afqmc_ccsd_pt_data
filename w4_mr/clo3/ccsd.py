from pyscf import gto, scf, cc

s = 1
nfrozen = 8
b = 'ccpvdz'
sym = False

atoms = '''
Cl       0.000000    0.000000    0.207132 
O        0.000000    1.400820   -0.146719 
O        1.213146   -0.700410   -0.146719 
O       -1.213146   -0.700410   -0.146719 
'''

mol = gto.M(atom=atoms, basis=b, spin=s, verbose=4, symmetry=sym)
mol.build()

mf = scf.RHF(mol)
mf.kernel()

stable = False
for i in range(10):
    print(f'mf stability test {i+1}')
    if not stable:
        mo_i, _, stable,_ = mf.stability(return_status=True)
        dm = mf.make_rdm1(mo_i,mf.mo_occ)
        mf.kernel(dm0=dm)
    elif stable:
        print(f'mf energy: {mf.e_tot}, stability {stable}')
        break

mycc = cc.CCSD(mf,frozen=nfrozen)
e = mycc.kernel()
et = mycc.ccsd_t()
print('RCCSD(T) energy = ', mycc.e_tot + et)

mf = scf.UHF(mol)
mf.kernel()

stable = False
for i in range(10):
    print(f'mf stability test {i+1}')
    if not stable:
        mo_i, _, stable,_ = mf.stability(return_status=True)
        dm = mf.make_rdm1(mo_i,mf.mo_occ)
        mf.kernel(dm0=dm)
    elif stable:
        print(f'mf energy: {mf.e_tot}, stability {stable}')
        break

mycc = cc.CCSD(mf,frozen=nfrozen)
e = mycc.kernel()
et = mycc.ccsd_t()
print('UCCSD(T) energy = ', mycc.e_tot + et)
