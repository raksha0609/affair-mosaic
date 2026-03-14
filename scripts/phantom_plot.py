# scripts/phantom_plot.py
"""
Visualize phantom evaluation results.
Reads results/phantom_eval_results.csv and plots key metrics.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_phantom_results():
    results_csv = "results/phantom_eval_results.csv"
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"{results_csv} not found. Run phantom_eval.py first.")

    df = pd.read_csv(results_csv)
    print("Loaded results:\n", df.head())

    # Normalize column names (strip whitespace etc.)
    df.columns = [c.strip() for c in df.columns]

    # Try to adapt to whichever column naming format exists
    possible_names = {
        "Rotation Error (°)": "Rotation Error (deg)",
        "Rotation Error (deg)": "Rotation Error (deg)",
        "Translation Error": "Translation Error",
        "Zoom Error": "Zoom Error"
    }

    # Map existing ones
    available_metrics = [name for name in possible_names if name in df.columns]

    if not available_metrics:
        print("[WARN] Could not find standard metric columns in CSV.")
        print("Found columns:", df.columns.tolist())
        return

    # Plot available metrics
    df_mean = df[available_metrics].mean()
    ax = df_mean.plot(kind="bar", color="skyblue", figsize=(7, 4))
    ax.set_title("Phantom Evaluation Metrics")
    ax.set_ylabel("Error / Deviation")
    plt.xticks(rotation=0)
    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/phantom_eval_metrics.png")
    print("✅ Plot saved at results/plots/phantom_eval_metrics.png")
    plt.show()

if __name__ == "__main__":
    plot_phantom_results()
