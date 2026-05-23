import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
def check_calibration(model, x_test, y_test,savename=None):
    y_prob = model.predict_proba(x_test)[:, 1]
    y_prob=np.array(y_prob)
    y_test=np.array(y_test)
    n_bins = 10
    bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-8   # include the minimum value
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    fraction_of_positives = np.zeros(n_bins)
    mean_predicted = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        bin_counts[b] = mask.sum()
        if mask.sum() > 0:
            fraction_of_positives[b] = y_test[mask].mean()
            mean_predicted[b] = y_prob[mask].mean()

    # Step 5: Inspect the bins
    for b in range(n_bins):
        print(f"Bin {b+1:2d} | n={int(bin_counts[b]):4d} | "
              f"mean_pred={mean_predicted[b]:.3f} | "
              f"frac_pos={fraction_of_positives[b]:.3f}")
    # ── 3. Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — Left: calibration curve —
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.plot(mean_predicted, fraction_of_positives,
            "o-", color="#2563eb", lw=2, ms=7, label="Model")
    for x, y, n in zip(mean_predicted, fraction_of_positives, bin_counts):
        ax.annotate(f"n={int(n)}", xy=(x, y),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=7.5, color="#555")

    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives",      fontsize=11)
    ax.set_title("Calibration Curve",           fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # — Right: predicted probability histogram —
    ax2 = axes[1]
    ax2.hist(y_prob, bins=30, color="#2563eb", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Count",                 fontsize=11)
    ax2.set_title("Distribution of Predicted Probabilities", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.show()