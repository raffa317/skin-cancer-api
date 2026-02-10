import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results():
    log_file = "training_history.csv"
    if not os.path.exists(log_file):
        print("Log file not found. Run training first.")
        return

    # Read data
    df = pd.read_csv(log_file)
    epochs = df['epoch']
    
    # Setup styles
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['val_acc'], label='Validation Acc', linewidth=2, color='#ff7f0e') # Orange
    # We didn't log train_acc in the simple edit, let's assume we might edit train.py later to include it
    # Or just plot what we have. If 'train_acc' is missing, skip it.
    if 'train_acc' in df.columns:
         plt.plot(epochs, df['train_acc'], label='Training Acc', linewidth=2, color='#1f77b4') # Blue
    
    plt.title('Training and Validation Accuracy Curves (AiDerm)', fontsize=14, weight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_accuracy_curve.png', dpi=300)
    print("Saved training_accuracy_curve.png")

    # 2. Loss Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['train_loss'], label='Training Loss', linewidth=2, color='#1f77b4') # Blue
    plt.plot(epochs, df['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e') # Orange
    
    plt.title('Training and Validation Loss Curves (AiDerm)', fontsize=14, weight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss_curve.png', dpi=300)
    print("Saved training_loss_curve.png")

if __name__ == "__main__":
    plot_training_results()
