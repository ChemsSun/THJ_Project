from pathlib import Path
from PIL import Image
import csv
import numpy as np

# Augmenter un peu si nécessaire, 0.1 (10%) est souvent un bon point de départ
WHITE_THRESHOLD = 0.1 

processed_folder = Path("processed")
datasets = ["data2_FSD", "data3_SFA", "data4_HGR"]

for ds_name in datasets:
    dataset = processed_folder / ds_name
    real_dir = dataset / "patches" / "real"
    mask_dir = dataset / "patches" / "mask"
    csv_file = dataset / f"{ds_name}_patches.csv"

    if not real_dir.exists():
        print(f"Saut de {ds_name} : dossier real non trouvé")
        continue

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Ajout de image_id pour faciliter la reconstruction plus tard
        writer.writerow(["image_id", "image_patch", "mask_patch", "label"])

        # Tri important pour garder l'ordre des indices lors de la lecture
        for real_patch in sorted(real_dir.glob("*.png")):
            mask_patch = mask_dir / real_patch.name

            if not mask_patch.exists():
                continue

            # Extraire l'ID de l'image (ex: '001' de '001_04.png')
            image_id = real_patch.stem.split('_')[0]

            # Traitement du masque
            mask = Image.open(mask_patch).convert("L")
            mask_array = np.array(mask)

            # Correction : Utilisation d'un seuil > 127 au lieu de == 255
            # pour gérer les artefacts de compression/redimensionnement
            white_ratio = np.mean(mask_array > 127)
            label = 1 if white_ratio >= WHITE_THRESHOLD else 0

            writer.writerow([
                image_id,
                str(real_patch.relative_to(dataset)),
                str(mask_patch.relative_to(dataset)),
                label
            ])

    print(f"{ds_name}: CSV créé avec succès.")