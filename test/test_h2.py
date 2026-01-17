from pyscf import gto, scf, cc

a = 2 # bond length in a cluster
d = 100 # distance between each cluster
na = 2  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
s = 0 # spin of each monomer
elmt = 'H'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g",spin=0, unit='bohr', verbose=4)
mol.build()

mf = scf.RHF(mol)
e = mf.kernel()

nfrozen = 0

#mycc = cc.CCSD(mf,frozen=nfrozen)
#mycc.kernel()

options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 10,
            'n_blocks': 600,
            'n_walkers': 300,
            'seed': 6,
            'walker_type': 'rhf',
            'trial': 'rhf',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True
            }

from ad_afqmc.prop_unrestricted import prop_unrestricted
prop_unrestricted.prep_afqmc(mf,options,chol_cut=1e-6)
prop_unrestricted.run_afqmc(options)
