from pyscf import gto, scf, cc
import os

d = 100
water = '''
O 0.0 0.0 0.0
O 0.0 0.0 1.20577
'''

m_list = [1]
for nc in m_list:
    atoms = ""
    for n in range(nc):
        shift = n*d
        atoms += f'O {0.0+shift} 0.0 0.0     \n'
        atoms += f'O {0.0+shift} 0.0 1.20577 \n'

    nfrozen = 2*nc
    spin = 2*nc
    mol = gto.M(atom=atoms, basis="sto6g", spin=spin, verbose=4)
    mol.build()

    mf = scf.UHF(mol)
    mf.kernel()
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mf.stability()

    options = {'n_eql': 4,
               'n_prop_steps': 50,
               'n_ene_blocks': 5,
               'n_sr_blocks': 10,
               'n_blocks': 100,
               'n_walkers': 200,
               'seed': 2,
               'walker_type': 'uhf',
               'trial': 'uhf',
               'dt':0.005,
               'free_projection':False,
               'ad_mode':None,
               'use_gpu': True,
               }

    from ad_afqmc.prop_unrestricted import prop_unrestricted
    prop_unrestricted.prep_afqmc(mf,options,chol_cut=1e-5,norb_frozen=nfrozen)
    prop_unrestricted.run_afqmc(options)
    os.system(f'mv afqmc.out afqmc_hf_m{nc}.out')
