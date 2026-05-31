import gradio as gr
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors,Draw

model=joblib.load("model.pkl")

DESCRIPTORS = ["MolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors",
    "NumRotatableBonds","NumAromaticRings","NumAliphaticRings",
    "NumSaturatedRings","FractionCSP3","HeavyAtomCount",
    "NumHeteroatoms","RingCount","BertzCT"]

def predict(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None:return "Invalid SMILES",None
    feats=[]
    for n in DESCRIPTORS:
        try:
            v=getattr(Descriptors,n)(mol)
            feats.append(v if np.isfinite(v) else 0)
        except:feats.append(0)
    pred=model.predict(np.array(feats).reshape(1,-1))[0]
    img=Draw.MolToImage(mol,size=(400,400))
    return f"Predicted Tg={pred:.1f}",img

demo=gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="SMILES",placeholder="*CC(*)c1ccccc1"),
    outputs=[gr.Textbox(label="Tg"),gr.Image(label="Structure")],
    title="Polymer Tg Predictor",
    examples=["*CC(*)c1ccccc1","*CC(*)C(=O)OC","*CC*","O=C(Oc1ccc(C)cc1)Oc2ccc(C)cc2"]
)
demo.launch()
