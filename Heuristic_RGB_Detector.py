import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

def detecteur_RGB(patch):

    # Conversion BGR vers RGB 
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(float)
    
    R_moyen = np.mean(patch_rgb[:, :, 0])
    G_moyen = np.mean(patch_rgb[:, :, 1])
    B_moyen = np.mean(patch_rgb[:, :, 2])
    
    # Application des 6 RÈGLES 
    regle1 = R_moyen > 95    
    regle2 = G_moyen > 40
    regle3 = B_moyen > 20
    regle4 = (max(R_moyen, G_moyen, B_moyen) - min(R_moyen, G_moyen, B_moyen)) > 15
    regle5 = abs(R_moyen - G_moyen) > 15
    regle6 = (R_moyen > G_moyen) and (G_moyen > B_moyen)
    
    # Décision FINALE 
    decision = 1 if (regle1 and regle2 and regle3 and regle4 and regle5 and regle6) else 0
    
    return decision, R_moyen, G_moyen, B_moyen

def application():
    DATABASES = ["data2_FSD", "data3_SFA", "data4_HGR"]
    BASE_PATH = Path("processed/")
    
    for db in DATABASES:
        print(f"\n{'='*40}")
        print(f"BASE : {db}")
        print(f"{'='*40}")
        
        db_folder = BASE_PATH / db
        csv_input = db_folder / f"{db}_patches.csv"
        patches_real_dir = db_folder / "patches" / "real"
        csv_output = db_folder / f"{db}_RGB_vf.csv"  
        
        if not csv_input.exists():
            print(f"ERREUR: {csv_input} introuvable")
            continue
        
        df = pd.read_csv(csv_input)
        df.columns = df.columns.str.strip()
        
        résultats = []
        
        # Traitement patch par patch
        for idx, ligne in df.iterrows():
            nom_patch = os.path.basename(ligne['image_patch'])
            chemin_patch = patches_real_dir / nom_patch
            
            img = cv2.imread(str(chemin_patch))
            if img is None:
                continue  # Skip si erreur
            
            # Application du DÉTECTEUR 
            decision, R, G, B = detecteur_RGB(img)
            
            résultats.append({
                "image_id": ligne['image_id'],
                "image_patch": ligne['image_patch'],
                "mask_patch": ligne['mask_patch'],
                "label_vrai": int(ligne['label']),
                "decision_RGB": decision,        # 0 ou 1 
                "moyenne_R": round(R, 2),        # Caractéristique R
                "moyenne_G": round(G, 2),        # Caractéristique G  
                "moyenne_B": round(B, 2)         # Caractéristique B
            })
            
        if résultats:
            df_final = pd.DataFrame(résultats)
            df_final.to_csv(csv_output, index=False)
            
            correct = (df_final['label_vrai'] == df_final['decision_RGB']).sum()
            précision = correct / len(df_final)
            
            print(f"\n TERMINÉ pour {db}")
            print(f"  Fichier: {csv_output}")
            print(f"  Précision RGB: {précision:.2%}")
            print(f"  Exemple ligne CSV:")
            print(f"    - Décision RGB: {df_final.iloc[0]['decision_RGB']}")
            print(f"    - Caractéristiques: R={df_final.iloc[0]['moyenne_R']}, "
                  f"G={df_final.iloc[0]['moyenne_G']}, B={df_final.iloc[0]['moyenne_B']}")

if __name__ == "__main__":
    application()