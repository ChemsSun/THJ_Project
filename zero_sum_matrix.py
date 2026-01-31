import pandas as pd
import numpy as np
import os
import shutil
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

class SkinZeroSumGameBuilder:
    def __init__(self, csv_base_path, image_base_path):
        """
        Initialize the Zero-Sum Game builder for skin detection.
        
        Args:
            csv_base_path: Path to directory containing classifier CSV results
            image_base_path: Path to directory containing image patches
        """
        rgb_path = os.path.join(csv_base_path, 'rgb_results_HGR.csv')
        ann_path = os.path.join(csv_base_path, 'ann_results_HGR.csv')
        hsv_path = os.path.join(csv_base_path, 'hsv_results_HGR.csv')
        
        print(f"--- Loading CSV files from {csv_base_path} ---")
        self.clfs = {
            'RGB': pd.read_csv(rgb_path),
            'ANN': pd.read_csv(ann_path),
            'HSV': pd.read_csv(hsv_path)
        }
        
        self.image_root = image_base_path
        
        # Feature columns for each strategy
        self.feature_map = {
            'RGB': ['moyenne_R', 'moyenne_G', 'moyenne_B'],
            'HSV': ['alpha_hsv', 'alpha_norm', 'beta_hsv'],
            'ANN': ['ann_logit', 'ann_conf', 'ann_energy']
        }
        
        # Prediction label columns
        self.pred_label_cols = {
            'RGB': 'label_rgb',
            'ANN': 'label_ann',
            'HSV': 'label_hsv'
        }
        
        self.gt_label_col = 'label'
        
        # Pre-convert features to numeric
        for strategy, cols in self.feature_map.items():
            for col in cols:
                self.clfs[strategy][col] = pd.to_numeric(
                    self.clfs[strategy][col], errors='coerce'
                )
        
        # Create indexed versions for fast lookup
        self.indexed_clfs = {
            name: df.set_index('image_patch') 
            for name, df in self.clfs.items()
        }
        
        print("✓ CSV files loaded and indexed successfully")
    
    def _get_strategy_skin_matrix(self, strategy_name):
        """
        Get the skin data matrix Ms for a given strategy.
        Contains features from patches where ALL classifiers agree on skin.
        
        Args:
            strategy_name: Name of the strategy ('RGB', 'ANN', or 'HSV')
            
        Returns:
            numpy array of skin features
        """
        # Merge all classifier results
        merged = (
            self.clfs['RGB']
            .merge(self.clfs['ANN'], on='image_patch', suffixes=('_rgb', '_ann'))
            .merge(self.clfs['HSV'], on='image_patch')
        )
        
        # Get patches where all classifiers agree on skin
        agreement = merged[
            (merged['label_rgb'] == 1) & 
            (merged['label_ann'] == 1) & 
            (merged['label_hsv'] == 1)
        ]
        
        cols = self.feature_map[strategy_name]
        
        if not agreement.empty:
            return agreement[cols].values.astype(np.float64)
        else:
            # Fallback: return small random matrix if no agreement
            print(f"⚠ Warning: No agreement patches for {strategy_name}, using fallback")
            return np.random.rand(10, len(cols))
    
    def calculate_correlation_distance(self, patch_features, Ms):
        """
        Calculate correlation distance between patch and skin data matrix.
        Implements the distance defined in Section 3.1.1(a) of the paper.
        
        Args:
            patch_features: Feature vector of the patch
            Ms: Skin data matrix for the strategy
            
        Returns:
            Correlation distance value
        """
        Ms = np.asarray(Ms, dtype=np.float64)
        patch_features = np.asarray(patch_features, dtype=np.float64)
        
        m, n = Ms.shape
        
        # Step 1: Compute mean and std
        mean = np.mean(Ms, axis=0)
        std = np.std(Ms, axis=0) + 1e-9  # Avoid division by zero
        
        # Step 2: Standardize the matrix
        Zs = (Ms - mean) / std
        
        # Step 3: Compute correlation matrix
        Rs = (1/m) * (Zs.T @ Zs)
        
        # Standardize the patch features
        Ts_bar = (patch_features - mean) / std
        
        # Step 4: Find closest vector in Zs
        diff = Zs - Ts_bar
        distances = np.linalg.norm(diff, axis=1)
        idx = np.argmin(distances)
        
        # Step 5: Compute Vs
        Vs = (Ts_bar - Zs[idx]).reshape(-1, 1)
        
        # Step 6: Compute correlation distance (Eq. 8)
        d = (Vs.T @ Rs @ Vs).item()
        
        # Return small positive value if distance is too small
        return d if d > 1e-12 else 1e-12
    
    def _get_spatial_neighbors(self, patch_path, all_patches_df):
        """
        Get spatial neighbors of a patch based on naming convention.
        Assumes patch naming follows pattern: imageX_patchY.ext
        
        Args:
            patch_path: Path to the patch
            all_patches_df: DataFrame containing all patches from same image
            
        Returns:
            List of neighbor patch paths
        """
        # Extract patch coordinates (this is a simplified version)
        # You may need to adjust based on your actual naming convention
        try:
            base_name = os.path.basename(patch_path)
            # Get all patches from same image as potential neighbors
            neighbors = all_patches_df['image_patch'].tolist()
            
            # Remove the patch itself
            neighbors = [n for n in neighbors if n != patch_path]
            
            return neighbors
        except Exception as e:
            print(f"⚠ Warning: Could not extract neighbors for {patch_path}: {e}")
            return []
    
    def calculate_utility(self, s1, s2, patch_path, Ms_dict, neighbor_paths):
        """
        Calculate utility function u(s1, s2) as defined in Eq. 11 of the paper.
        
        Args:
            s1: Strategy name for skin player
            s2: Strategy name for non-skin player
            patch_path: Path to the conflict patch
            Ms_dict: Dictionary of skin matrices for each strategy
            neighbor_paths: List of neighbor patch paths
            
        Returns:
            Utility value (beta - alpha)
        """
        try:
            # Get patch features for s1
            f_p1 = self.indexed_clfs[s1].loc[patch_path, self.feature_map[s1]].values
            d_p1 = self.calculate_correlation_distance(f_p1, Ms_dict[s1])
            
            # Get SKIN neighbors according to s1 (for alpha calculation)
            skin_neighbors_s1 = [
                n for n in neighbor_paths 
                if n in self.indexed_clfs[s1].index 
                and self.indexed_clfs[s1].loc[n, self.pred_label_cols[s1]] == 1
            ]
            
            # Calculate alpha (Eq. 9)
            if skin_neighbors_s1:
                d_neighbors_1 = [
                    self.calculate_correlation_distance(
                        self.indexed_clfs[s1].loc[n, self.feature_map[s1]].values,
                        Ms_dict[s1]
                    ) for n in skin_neighbors_s1
                ]
                
                # min |ds1,P - ds1,v|
                min_diff = min([abs(d_p1 - dn) for dn in d_neighbors_1])
                # Σ ds1,v
                sum_distances = sum(d_neighbors_1)
                # alpha = min_diff × sum_distances
                alpha = min_diff * sum_distances
            else:
                alpha = 0
            
            # Get patch features for s2
            f_p2 = self.indexed_clfs[s2].loc[patch_path, self.feature_map[s2]].values
            d_p2 = self.calculate_correlation_distance(f_p2, Ms_dict[s2])
            
            # Get NON-SKIN neighbors according to s2 (for beta calculation)
            nonskin_neighbors_s2 = [
                n for n in neighbor_paths 
                if n in self.indexed_clfs[s2].index 
                and self.indexed_clfs[s2].loc[n, self.pred_label_cols[s2]] == 0
            ]
            
            # Calculate beta (Eq. 10)
            if nonskin_neighbors_s2:
                d_neighbors_2 = [
                    self.calculate_correlation_distance(
                        self.indexed_clfs[s2].loc[n, self.feature_map[s2]].values,
                        Ms_dict[s2]
                    ) for n in nonskin_neighbors_s2
                ]
                
                inv_d_p2 = 1 / d_p2
                inv_d_neighbors = [1/dn for dn in d_neighbors_2]
                
                # min |1/ds2,P - 1/ds2,v|
                min_diff = min([abs(inv_d_p2 - inv_dn) for inv_dn in inv_d_neighbors])
                # Σ 1/ds2,v
                sum_inv_distances = sum(inv_d_neighbors)
                # beta = min_diff × sum_inv_distances
                beta = min_diff * sum_inv_distances
            else:
                beta = 0
            
            # Utility u(s1, s2) = beta - alpha (Eq. 11)
            return beta - alpha
            
        except Exception as e:
            print(f"⚠ Error calculating utility for {patch_path}: {e}")
            return 0.0
    
    def generate_matrices(self, output_dir="/kaggle/working/zero_sum_conflict_matrices"):
        """
        Generate zero-sum game payoff matrices for all conflict patches.
        
        Args:
            output_dir: Directory to save the resulting CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Pre-compute skin matrices for all strategies
        print("\n--- Computing skin data matrices (Ms) for each strategy ---")
        Ms_dict = {}
        for name in self.clfs.keys():
            print(f"Computing Ms for {name}...")
            Ms_dict[name] = self._get_strategy_skin_matrix(name)
            print(f"  Ms shape: {Ms_dict[name].shape}")
        
        all_patches = self.clfs['RGB']['image_patch'].unique()
        conflict_count = 0
        saved_count = 0
        
        print(f"\n--- Processing {len(all_patches)} patches ---")
        
        for p_path in tqdm(all_patches, desc="Building Conflict Matrices"):
            full_image_path = os.path.join(self.image_root, p_path)
            
            # Get labels from each classifier
            labels = {}
            for name in self.clfs:
                if p_path in self.indexed_clfs[name].index:
                    labels[name] = self.indexed_clfs[name].loc[
                        p_path, self.pred_label_cols[name]
                    ]
            
            # Determine strategies for each player
            S1 = [k for k, v in labels.items() if v == 1]  # Skin player strategies
            S2 = [k for k, v in labels.items() if v == 0]  # Non-skin player strategies
            
            # Skip if not a conflict patch (all agree or only one classifier)
            if not (S1 and S2):
                continue
            
            conflict_count += 1
            
            # Get ground truth label and image_id
            try:
                real_label = self.indexed_clfs['ANN'].loc[p_path, self.gt_label_col]
                img_id = self.indexed_clfs['RGB'].loc[p_path, 'image_id']
            except KeyError:
                continue
            
            # Get neighbors from same image
            neighbors_df = self.clfs['RGB'][self.clfs['RGB']['image_id'] == img_id]
            neighbor_paths = self._get_spatial_neighbors(p_path, neighbors_df)
            
            # Build payoff matrix A
            A = np.zeros((len(S1), len(S2)))
            
            for i, s1 in enumerate(S1):
                for j, s2 in enumerate(S2):
                    A[i, j] = self.calculate_utility(
                        s1, s2, p_path, Ms_dict, neighbor_paths
                    )
            
            # Save matrix to CSV
            df_A = pd.DataFrame(A, index=S1, columns=S2)
            df_A.insert(0, "region_path", full_image_path)
            df_A.insert(1, "real_label", real_label)
            
            # Create clean filename
            clean_name = os.path.basename(p_path).replace('.', '_')
            output_path = os.path.join(output_dir, f"zero_sum_{clean_name}.csv")
            df_A.to_csv(output_path, index=True)
            saved_count += 1
        
        print(f"\n Finished processing:")
        print(f"   Total patches: {len(all_patches)}")
        print(f"   Conflict patches: {conflict_count}")
        print(f"   Matrices saved: {saved_count}")
        print(f"   Output directory: {output_dir}")


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    CSV_PATH = '/kaggle/input/csv-files'
    IMG_PATH = '/kaggle/input/skin-patches-dataset/data4_HGR'
    OUT_PATH = '/kaggle/working/zero_sum_conflict_matrices'
    
    print("="*70)
    print("ZERO-SUM GAME THEORY MODEL FOR SKIN SEGMENTATION")
    print("Based on Dahmani et al. (2020)")
    print("="*70)
    
    # Initialize builder
    builder = SkinZeroSumGameBuilder(CSV_PATH, IMG_PATH)
    
    # Generate conflict matrices
    builder.generate_matrices(OUT_PATH)
    
    # Create zip archive
    print("\n--- Creating ZIP archive ---")
    shutil.make_archive(
        '/kaggle/working/conflict_results', 
        'zip', 
        OUT_PATH
    )
    print("Archive created: conflict_results.zip")
    print("\n" + "="*70)