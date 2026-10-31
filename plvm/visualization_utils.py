"""Training-curve plot shared by the training scripts."""

import os
from matplotlib import pyplot as plt
import logging
import traceback


def plot_and_save_training_curve(
    log_dir, model_name, losses, val_losses, num_epochs, args, plot_ylims=None
):
    """Save the loss curves of one run to ``<log_dir>/<model_name>/plots/``.

    ``val_losses`` may be sparser than ``losses``; its points are spaced evenly over the
    epochs.  ``args`` supplies the hyperparameter box; ``plot_ylims`` is ``(ymin, ymax)``.
    """
    # Create directory for model plots if it does not exist
    model_log_dir = os.path.join(log_dir, f"{model_name}/plots")
    os.makedirs(model_log_dir, exist_ok=True)
    try:
        # Plot training and validation loss
        plt.figure(figsize=(14, 8))
        epochs = range(1, len(losses) + 1)
        # Validation may run every few epochs; space its points accordingly.
        interval = max(1, len(losses) // len(val_losses))
        val_epochs = range(1, len(losses) + 1, interval)

        plt.plot(
            epochs,
            losses,
            label=f"Training Loss (Final: {losses[-1]:.4f})",
            color="blue",
            linestyle="-",
            linewidth=2,
        )
        plt.plot(
            val_epochs,
            val_losses,
            label=f"Validation Loss (Final: {val_losses[-1]:.4f})",
            color="red",
            linestyle="--",
            linewidth=2,
        )
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        if plot_ylims:
            plt.ylim(plot_ylims[0], plot_ylims[1])
        plt.title(f"Training and Validation Loss for {model_name}", fontsize=16)
        plt.legend()
        plt.grid(True)

        # Create hyperparameter text box
        hyperparam_text = f"""Key Hyperparameters:\n
Task: {getattr(args, "task_label", "N/A")}\n
Model: {getattr(args, "model_type", "N/A")}\n
Loss Type: {getattr(args, "loss_type", "N/A")}\n
Feature Encoding: {getattr(args, "feature_encoding", "N/A")}\n
Learning Rate: {getattr(args, "learning_rate", "N/A")}\n
Batch Size: {getattr(args, "batch_size", "N/A")}\n
Hidden Size: {getattr(args, "hidden_size", "N/A")}\n
Num Layers: {getattr(args, "num_layers", "N/A")}\n
Seq Length: {getattr(args, "seq_length", "N/A")}\n
Prediction Horizon: {getattr(args, "prediction_horizon", "N/A")}\n
Max Norm: {getattr(args, "max_norm", "N/A")}\n"""

        # Add text box to the plot
        plt.text(
            0.02,
            0.98,
            hyperparam_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save the plot
        plot_file = os.path.join(model_log_dir, f"{model_name}_training_curve.pdf")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Training curve plot saved as {plot_file}.")
    except Exception:
        # log error traceback
        logging.error(f"Error plotting training curve for {model_name}: {traceback.format_exc()}")
