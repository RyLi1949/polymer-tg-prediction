print("代码开始运行了")
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("polymer_tg.csv.csv")

#1.Print basic info
print("Shape:",df.shape)
print("Columns:",df.columns.tolist())
print(df.head())

#2.Plot Tg distribution
plt.figure(figsize=(8,4))
plt.hist(df["labels.Exp_Tg(K)"],bins=50)
plt.xlabel("Tg" )
plt.ylabel("count")
plt.title("Tg distribution")
plt.savefig("tg_dist.png",dpi=120,bbox_inches="tight")
plt.show()

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

DESCRIPTORS=["MolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors","NumRotatableBonds","NumAromaticRings","NumAliphaticRings","NumSaturatedRings","FractionCSP3","HeavyAtomCount","NumHeteroatoms","RingCount","BertzCT"]
def features(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None:return None
    out=[]
    for name in DESCRIPTORS:
        try:
            v=getattr(Descriptors,name)(mol)
            out.append(v if np.isfinite(v) else(0))
        except:out.append(0)
    return out

X_list,y_list=[],[]
for _,row in df.iterrows():
    f=features(row["PSMILES"])
    if f is not None:
        X_list.append(f)
        y_list.append(row["labels.Exp_Tg(K)"])

X=np.array(X_list);y=np.array(y_list)    
print("X:",X.shape,"y:",y.shape)
np.save("X.npy",X);np.save("y.npy",y)

























