import cv2
from pathlib import Path

# Paramètres
DATASETS = ["data2_FSD", "data3_SFA", "data4_HGR"]
IMAGE_SIZE = 256       # Taille finale des images
PATCH_SIZE = 16       # Taille des patches
BRIGHTNESS_FACTOR = 0.9

def adjust_brightness(img, factor=0.9):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

for dataset in DATASETS:
    print(f"Traitement du dataset {dataset}...")

    RAW_REAL_DIR = Path(f"{dataset}/real")
    RAW_MASK_DIR = Path(f"{dataset}/mask")

    # Dossiers de sortie
    PROC_REAL_DIR = Path(f"processed/{dataset}/real")
    PROC_MASK_DIR = Path(f"processed/{dataset}/mask")
    PATCH_REAL_DIR = Path(f"processed/{dataset}/patches/real")
    PATCH_MASK_DIR = Path(f"processed/{dataset}/patches/mask")

    for d in [PROC_REAL_DIR, PROC_MASK_DIR, PATCH_REAL_DIR, PATCH_MASK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    real_files = sorted(RAW_REAL_DIR.glob("*.*"))
    mask_files = sorted(RAW_MASK_DIR.glob("*.*"))

    for idx, (r_file, m_file) in enumerate(zip(real_files, mask_files), start=1):
        img = cv2.imread(str(r_file))
        mask = cv2.imread(str(m_file), cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"Fichier manquant : {r_file} ou {m_file}")
            continue

        # Redimensionner l'image et le masque
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # Ajuster uniquement la luminosité de l'image
        img = adjust_brightness(img, BRIGHTNESS_FACTOR)

        # Sauvegarder l'image et le masque complets
        img_name = f"{idx:03d}.png"
        mask_name = f"{idx:03d}.png"
        cv2.imwrite(str(PROC_REAL_DIR / img_name), img)
        cv2.imwrite(str(PROC_MASK_DIR / mask_name), mask)

        # Découper en patches 16x16
        patch_idx = 0
        for y in range(0, IMAGE_SIZE, PATCH_SIZE):
            for x in range(0, IMAGE_SIZE, PATCH_SIZE):
                patch_img = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_mask = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                patch_name = f"{idx:03d}_{patch_idx:03d}.png"
                cv2.imwrite(str(PATCH_REAL_DIR / patch_name), patch_img)
                cv2.imwrite(str(PATCH_MASK_DIR / patch_name), patch_mask)
                patch_idx += 1

print("Prétraitement 16x16 terminé avec masque intact !")
