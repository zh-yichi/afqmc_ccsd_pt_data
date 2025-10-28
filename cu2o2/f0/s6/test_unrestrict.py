options = {'n_eql': 1,
           'n_prop_steps': 10,
            'n_ene_blocks': 1,
            'n_sr_blocks': 1,
            'n_blocks': 1,
            'n_walkers': 1,
            'seed': 2,
            'walker_type': 'uhf',
            'trial': 'uccsd_pt2_ad', # ccsd_pt,ccsd_pt_ad,ccsd_pt2_ad, uccsd_pt, uccsd_pt_ad, uccsd_pt2_ad
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True,
            }

from jax import config
config.update("jax_enable_x64", True)

from ad_afqmc import config as config
config.afqmc_config = {"use_gpu": True}

from ad_afqmc import pyscf_interface, run_afqmc
from ad_afqmc.prop_unrestricted import prop_unrestricted
prop_unrestricted.prep_afqmc(mycc,options,chol_cut=1e-6)
prop_unrestricted.run_afqmc(options)
