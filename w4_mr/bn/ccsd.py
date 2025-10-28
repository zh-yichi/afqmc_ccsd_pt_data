from pyscf import gto, scf, cc

s = 0
nfrozen = 0
b = 'ccpvdz'

atoms = '''
B        0.000000    0.000000   -0.748417
N        0.000000    0.000000    0.534583
'''

mol = gto.M(atom=atoms, basis=b, spin=s, verbose=4)
mol.build()

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
print('CCSD(T) energy = ', mycc.e_tot + et)
