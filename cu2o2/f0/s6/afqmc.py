from ad_afqmc.ccsd_pt import sample_uccsd_pt2

#from jax import config

#config.update("jax_enable_x64", True)

options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 5,
            'n_sr_blocks': 5,
            'n_blocks': 10,
            'n_walkers': 1,
            'seed': 720017,
            'walker_type': 'uhf',
            'trial': 'uccsd_pt2_ad', # ccsd_pt,ccsd_pt_ad,ccsd_pt2_ad, ucisd
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True,
            }

sample_uccsd_pt2.run_afqmc(options)
