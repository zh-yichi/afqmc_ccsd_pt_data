import json
from pyscf import gto, scf, cc
import numpy as np

df=json.load(open("trail.json"))

re={'ScO':1.668,
    'TiO':1.623,
    'VO':1.591,
    'CrO':1.621,
    'MnO':1.648,
    'FeO':1.616,
    'CuO':1.725,
    }

tmo_spins={'ScO':1,'TiO':2,'VO':3,'CrO':4,'MnO':5,'FeO':4,'CuO':1}
tmo_n3d={'Sc':(1,0),'Ti':(2,0),'V':(3,0),'Cr':(4,0),'Mn':(5,0),'Fe':(5,1),'Cu':(5,4)}
tmo_n4s={'Sc':(0,0),'Ti':(0,0),'V':(0,0),'Cr':(0,0),'Mn':(0,0),'Fe':(0,0),'Cu':(0,0)}

atom_spins={'O':2,'Sc':1,'Ti':2,'V':3,'Cr':6,'Mn':5,'Fe':4,'Cu':1}
atom_n3d={'Sc':(1,0),'Ti':(2,0),'V':(3,0),'Cr':(5,0),'Mn':(5,0),'Fe':(5,1),'Cu':(5,5)}
atom_n4s={'Sc':(1,1),'Ti':(1,1),'V':(1,1),'Cr':(1,0),'Mn':(1,1),'Fe':(1,1),'Cu':(1,0) }

for basis in ['vtz']:
  for el in ['Sc']:
    for charge in [0]:
      molname=el+'O'
      mol=gto.Mole()
      mol.ecp={}
      mol.basis={}
      for e in [el,'O']:
        mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
        mol.basis[e]=gto.basis.parse(df[e][basis])
      mol.charge=charge
      mol.spin=tmo_spins[molname]
      mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(el,re[molname]),verbose=4)

      aos = mol.ao_labels()
      orb2s = np.array([i for i, x in enumerate(aos) if "2s" in x])
      orb2p = np.array([i for i, x in enumerate(aos) if "2p" in x])
      orb3s = np.array([i for i, x in enumerate(aos) if "3s" in x])
      orb3p = np.array([i for i, x in enumerate(aos) if "3p" in x])
      orb4s = np.array([i for i, x in enumerate(aos) if "4s" in x])
      orb3d = np.array([i for i, x in enumerate(aos) if "3d" in x])
      dm0 = np.zeros((2, mol.nao, mol.nao))
      # costumize for each tmo
      double_occ = np.concatenate([orb2s,orb2p,orb3s,orb3p,orb4s])
      single_occ = orb3d[0]
      for i in double_occ:
          dm0[0, i, i] = 1.0
          dm0[1, i, i] = 1.0
      for i in single_occ:
          dm0[0, i, i] = 1.0

      mf = scf.RHF(mol)
      mf.chkfile = el + basis + str(charge) + ".chk"
      mo1 = None
      mf.level_shift = 0.5
      mf.kernel(dm0)
      mo1 = mf.stability()[0]
      mf = mf.newton().run(mo1, mf.mo_occ)
      mo1 = mf.stability()[0]
      mf = mf.newton().run(mo1, mf.mo_occ)
      mo1 = mf.stability()[0]
      mf = mf.newton().run(mo1, mf.mo_occ)
      mf.stability()

      umf = scf.UHF(mol)
      mo_occ = [
              np.array([1 for i in range(mol.nelec[0])] + [0 for i in range(mol.nao-mol.nelec[0])]),
              np.array([1 for i in range(mol.nelec[1])] + [0 for i in range(mol.nao-mol.nelec[1])]),
              ]
      dm0 = umf.make_rdm1([mo1, mo1], mo_occ)
      umf.level_shift = 0.5
      umf.kernel(dm0)
      mo1 = umf.stability()[0]
      umf = umf.newton().run(mo1, umf.mo_occ)
      mo1 = umf.stability()[0]
      umf = umf.newton().run(mo1, umf.mo_occ)

      norb_frozen = 0
      mycc = cc.CCSD(umf)
      mycc.frozen = norb_frozen
      mycc.run()
      et = mycc.ccsd_t()
      print(f"CCSD(T) energy: {mycc.e_tot + et}")