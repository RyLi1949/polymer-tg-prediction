print("ok!")
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_absolute_error

X=np.load("X.npy")
y=np.load("y.npy")

kf=KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
r2s,maes=[],[]

for fold,(tr,te) in enumerate(kf.split(X)):
    model=XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X[tr],
              y[tr],
              )    
    pred=model.predict(X[te])
    r2=r2_score(y[te],pred)
    mae=mean_absolute_error(y[te],pred)
    r2s.append(r2);maes.append(mae)
    print(f"Fold{fold+1}:R^2={r2:.4f},MAE={mae:.2f}")

print(f"\nR^2={np.mean(r2s):.4f}+/-{np.std(r2s):.4f}")  
print(f"MAE={np.mean(maes):.2f}+/-{np.std(maes):.2f}")

import matplotlib.pyplot as plt
import joblib
DESCRIPTORS=["MolWt","MolLogP","TPSA",
             "NumHAcceptors","NumHDonors",
             "NumRotatableBonds","NumAromaticRings",
             "NumAliphaticRings","NumSaturatedRings",
             "FractionCSP3","HeavyAtomCount",
             "NumHeteroatoms","RingCount","BertzCT",
             ]

#Train final model on all data
final=XGBRegressor(n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        )
final.fit(X,y)
joblib.dump(final,"model.pkl")
print("Saved to model.pkl")

importances=final.feature_importances_
order=np.argsort(importances)[::-1]
plt.figure(figsize=(8,5))
plt.barh([DESCRIPTORS[i] for i in order],importances[order],color="#1F5641")
plt.xlabel("Importance")
plt.title("What drives Tg?")
plt.tight_layout()
plt.savefig("importance.png",dpi=120)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

X_tr,X_te,y_tr,y_te=train_test_split(
    X,y,test_size=0.2,random_state=42)
m=XGBRegressor(n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        )
m.fit(X_tr,y_tr)
pred=m.predict(X_te)

plt.figure(figsize=(6,6))
plt.scatter(y_te,pred,alpha=0.4,s=15,color="#1F5641")
plt.plot([y.min(),y.max()],[y.min(),y.max()],"r--",lw=1.5)
plt.xlabel("True Tg")
plt.ylabel("Predicted Tg")
plt.title(f"R^2={r2_score(y_te,pred):.3f}")
plt.tight_layout()
plt.savefig("pred_vs_true.png",dpi=120)
plt.show()
print("ok!!")

