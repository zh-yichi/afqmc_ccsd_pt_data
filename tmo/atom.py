import json
from pyscf import gto, scf, cc
import numpy as np

df=json.load(open("../trail.json"))

spins={'O':2,'Sc':1,'Ti':2,'V':3,'Cr':6,'Mn':5,'Fe':4,'Cu':1}


nd={'O':(0,0), 'Sc':(1,0),'Ti':(2,0),'V':(3,0),'Cr':(5,0),'Mn':(5,0),'Fe':(5,1),
     'Cu':(5,5) }

for basis in ['vtz']:
  for el in ['Fe']:
    for charge in [0]:
      mol=gto.Mole()
      mol.ecp={}
      mol.basis={}
      mol.ecp[el]=gto.basis.parse_ecp(df[el]['ecp'])
      mol.basis[el]=gto.basis.parse(df[el][basis])
      mol.charge=charge
      #mol.symmetry='D4h'
      if el == 'Cr' or el == 'Cu':
        mol.spin=spins[el]-charge
      else:
        mol.spin=spins[el]+charge

      mol.build(atom="%s 0. 0. 0."%el,verbose=4)

      aos = mol.ao_labels()
      s3_orbs = np.array([i for i, x in enumerate(aos) if "3s" in x])
      p3_orbs = np.array([i for i, x in enumerate(aos) if "3p" in x])
      s4_orbs = np.array([i for i, x in enumerate(aos) if "4s" in x])
      d3_orbs = np.array([i for i, x in enumerate(aos) if "3d" in x])
      dm0 = np.zeros((2, mol.nao, mol.nao))
      double_occ = np.concatenate([s3_orbs, p3_orbs, s4_orbs, d3_orbs[:1]])
      single_occ = d3_orbs[1:]
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