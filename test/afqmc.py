options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 5,
            'n_blocks': 10,
            'n_walkers': 100,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'uccsd_pt2_ad', # ccsd_pt,ccsd_pt_ad,ccsd_pt2_ad, ucisd
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True
            }

# from ad_afqmc import pyscf_interface
from ad_afqmc.ccsd_pt import sample_uccsd_pt2
# pyscf_interface.prep_afqmc(mycc,options,chol_cut=1e-5)
sample_uccsd_pt2.run_afqmc(options)

