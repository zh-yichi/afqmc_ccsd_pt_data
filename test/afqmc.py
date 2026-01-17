options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 5,
            'n_blocks': 100,
            'n_walkers': 400,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'rhf',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True
            }

from ad_afqmc.prop_unrestricted import prop_unrestricted
prop_unrestricted.run_afqmc(options)

