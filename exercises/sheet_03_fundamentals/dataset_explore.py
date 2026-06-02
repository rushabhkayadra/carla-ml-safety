import sys
import os
import pandas as pd

# Append root path context to allow native package discovery of src tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def execute_dataset_audit(train_csv_path, test_csv_path):
    """
    Performs empirical data audit on CARLA input pipelines to map class imbalances.
    """
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        print("[CRITICAL] Metadata indexes missing. Ensure paths match environmental targets.")
        return

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    print("=========================================================================")
    print("                     CARLA DATASET METRIC AUDIT                          ")
    print("=========================================================================")
    print(f"Total Volumetric Count (Training Split) : {len(train_df)} samples")
    print(f"Total Volumetric Count (Testing Split)  : {len(test_df)} samples\n")
    
    labels = ['pedestrian', 'vehicle', 'traffic_light']
    
    print("--- Categorical Class Distribution Balance ---")
    for label in labels:
        pos_train = train_df[label].sum()
        neg_train = len(train_df) - pos_train
        ratio = pos_train / neg_train if neg_train > 0 else float('inf')
        print(f"Target Cluster [{label.upper()}]:")
        print(f"  -> Positive Instances (Present = 1) : {pos_train}")
        print(f"  -> Negative Instances (Absent = 0)  : {neg_train}")
        print(f"  -> Class Imbalance Ratio (1:0)       : {ratio:.4f}")
        
    print("\n--- Structural Co-occurrence Interaction Matrix (Training) ---")
    co_occurrence = train_df[labels].corr()
    print(co_occurrence)
    print("=========================================================================")

if __name__ == "__main__":
    # Modify placeholder targets to line up with physical storage locations
    execute_dataset_audit("../../data/train_metadata.csv", "../../data/test_metadata.csv")