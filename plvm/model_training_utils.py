"""Shared model code and training utilities for the task encoders and the PLVM.

Contents:

- sequence models for the frozen task encoders (``TransformerModel``, ``LSTMModel``) and
  their training loop (``train_task_specific_model``), save/load helpers;
- embedding extraction from frozen encoders (``MultiTaskEmbedModel``,
  ``precompute_embeddings``);
- PLVM heads (fuser to latent, latent decoders, classification decoder, MLP head) and the
  training loop on precomputed embeddings (``train_embedding_hierarchical_model``);
- naive baselines, run registries (``model_performance.csv``) and diagnostic plots written
  next to each run.
"""

from datetime import datetime
import logging
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import time
import random
import traceback
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Sampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler
import torch.cuda.amp as amp
import torch.nn.functional as F

from umap.umap_ import UMAP
import csv, json
import seaborn as sns
from typing import List, Dict, Optional

from plvm import artifact_paths

COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable scaled-dot-product attention backends (FlashAttention when available) for Transformer layers
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logging.info("Enabled Flash/SDP backends for Transformer attention")
    except Exception as e:
        logging.warning(f"Could not enable Flash/SDP attention backends: {e}")
PAD_VALUE = -20.0
FILL_MISSING_VALUE = -10.0


def build_padding_mask(
    batch: torch.Tensor,
    fill_value: float = FILL_MISSING_VALUE,
    extra_fill_values: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Return a boolean mask where True indicates padded timesteps.
    Padding is assumed to be filled with one or more sentinel values.
    """
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch)
    if extra_fill_values is None:
        extra_fill_values = [PAD_VALUE] if PAD_VALUE != fill_value else []
    if isinstance(fill_value, (list, tuple, set)):
        fill_values = list(fill_value)
    else:
        fill_values = [fill_value]
    for value in extra_fill_values:
        if value not in fill_values:
            fill_values.append(value)

    is_fill = torch.zeros(batch.shape[:2], dtype=torch.bool, device=batch.device)
    for value in fill_values:
        fill = torch.as_tensor(value, device=batch.device, dtype=batch.dtype)
        is_fill |= torch.isclose(batch, fill).all(dim=2)
    is_nan = torch.isnan(batch).any(dim=2)
    return is_fill | is_nan


class LSTMModel(nn.Module):
    """LSTM sequence model with a per-timestep linear output layer."""

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        """Build an LSTM with ``num_layers`` layers and one output per timestep."""
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, padding_mask=None):
        """Return per-timestep outputs of shape (batch, seq_len, output_size).

        ``padding_mask`` is ignored; it keeps the call signature of :class:`TransformerModel`.
        """
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # One output per time step.
        return self.fc(out)


class TransformerModel(nn.Module):
    """Causal transformer encoder with a per-timestep linear output layer (the task encoders)."""

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, nhead=8, dropout=0.1):
        """Build the encoder; ``hidden_size`` is the embedding dimension used by the PLVM."""
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nhead = nhead

        # Store hyperparameters in a dictionary for better retrieval
        self.hyperparameters = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "num_layers": num_layers,
            "nhead": nhead,
            "dropout": dropout,
        }

        # Embedding layer to project input to hidden dimension
        self.embedding = nn.Linear(input_size, hidden_size)

        # Positional encoding to add positional information
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        # Use batch-first attention to match input layout and avoid nested tensor warnings.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer to map transformer output to output space
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, padding_mask=None):
        """Return per-timestep outputs of shape (batch, seq_len, output_size)."""
        out = self.get_embedding(x, padding_mask=padding_mask)  # Project input to hidden size

        # Pass the output of each time step through the fully connected layer
        out = self.fc(out)  # Back to (batch_size, seq_length, output_size)

        return out

    def get_embedding(self, x, padding_mask=None):
        """Return the causal encoder states of shape (batch, seq_len, hidden_size)."""
        # Project input to hidden size
        x = self.embedding(x)

        # Keep batch-first layout for transformer; positional encoder expects seq-first.
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Create a causal mask
        seq_length = x.size(1)
        device = x.device
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device), diagonal=1
        ).bool()

        # Pass through the Transformer encoder with the causal mask
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to sequence-first inputs."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """Precompute the encoding table for sequences of up to ``max_len`` steps."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Standard sinusoidal encoding: sine on even feature indices, cosine on odd ones,
        # with wavelengths growing geometrically up to 10000 * 2 * pi.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape (max_len, 1, d_model), broadcast over the batch of a sequence-first input.
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add the positional encoding to ``x`` of shape (seq_len, batch, d_model)."""
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def generate_model_name():
    """Return a random lowercase run name ``<noun>-<verb>-<last 4 digits of Unix time>``."""
    nouns = [
        "Falcon",
        "Panther",
        "Hawk",
        "Tiger",
        "Eagle",
        "Phoenix",
        "Dragon",
        "Leopard",
        "Knight",
        "Viper",
        "Griffin",
        "Wolf",
        "Raven",
        "Cheetah",
        "Lion",
        "Ocelot",
        "Cobra",
        "Jaguar",
        "Serpent",
        "Bear",
        "Stallion",
        "Mustang",
        "Raptor",
        "Puma",
        "Tornado",
        "Cyclone",
        "Tsunami",
        "Thunder",
        "Blizzard",
        "Tempest",
        "Shark",
        "Orca",
        "Piranha",
        "Barracuda",
        "Manta",
        "Kraken",
        "Hydra",
        "Minotaur",
        "Centaur",
        "Pegasus",
        "Basilisk",
        "Chimera",
        "Sphinx",
        "Wyvern",
        "Cerberus",
        "Golem",
        "Yeti",
        "Mammoth",
        "Saber",
        "Scorpion",
        "Anaconda",
        "Komodo",
        "Gryphon",
        "Lynx",
        "Cougar",
        "Wolverine",
        "Badger",
        "Fox",
        "Hound",
        "Mongoose",
    ]

    verbs = [
        "Strike",
        "Run",
        "Fly",
        "Roar",
        "Charge",
        "Blaze",
        "Dash",
        "Glide",
        "Soar",
        "Dive",
        "Sprint",
        "Pounce",
        "Surge",
        "Flash",
        "Zoom",
        "Leap",
        "Rage",
        "Bolt",
        "Thrust",
        "Sweep",
        "Ascend",
        "Rush",
        "Storm",
        "Lunge",
        "Blast",
        "Thrive",
        "Ambush",
        "Pierce",
        "Unleash",
        "Ignite",
        "Climb",
        "Swoop",
        "Gallop",
        "Hurdle",
        "Skim",
        "Skate",
        "Slide",
        "Trek",
        "Vault",
        "Whirl",
        "Whiz",
        "Whip",
        "Whisk",
        "Whirlwind",
        "Zigzag",
        "Zoom",
        "Barrel",
        "Bound",
        "Burst",
        "Cannon",
        "Catapult",
        "Charge",
        "Chase",
        "Cruise",
        "Dart",
        "Dash",
        "Flee",
        "Gallop",
        "Hasten",
        "Hurtle",
        "Jet",
        "Lunge",
        "Plunge",
        "Race",
        "Rocket",
        "Rush",
        "Scamper",
        "Scurry",
        "Shoot",
        "Skim",
    ]

    # Generate a 4-digit time-based number (e.g., last 4 digits of current time in seconds)
    timestamp = int(time.time())  # Current time in seconds
    number = timestamp % 10000  # Get the last 4 digits
    # Create a new random generator instance
    local_random = random.Random()

    # Use the local_random instance to make choices
    noun = local_random.choice(nouns)
    verb = local_random.choice(verbs)

    model_name = f"{noun}-{verb}-{number:04d}".lower()  # Ensure the number is always 4 digits
    return model_name


def get_current_time_str():
    """Return the current time as ``"YYYY-mm-dd_HH-MM-SS"``."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class MultiTaskEmbedModel(nn.Module):
    """Embed each task's sequence with its frozen task encoder and fuse the task embeddings."""

    def __init__(
        self,
        models,
        max_len,
        fusion_type="concat",
        task_specific_hidden_size=128,
        pool_kernel_size=64,
        pool_stride=32,
        embeddings_to_use="last_only",
    ):
        super(MultiTaskEmbedModel, self).__init__()
        self.models = models
        for model in self.models:
            model.eval()
        self.k = len(models)
        self.max_len = max_len
        self.fusion_type = fusion_type
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.task_specific_hidden_size = task_specific_hidden_size
        self.embeddings_to_use = embeddings_to_use

        # Output size for features only (without validity indicators)
        # Each model produces a tensor of shape (batch_size, task_embedding_dim)
        # After concatenating k models, total size is task_embedding_dim * k
        self.output_size = self._get_task_embedding_dim() * self.k

    def _get_task_embedding_dim(self) -> int:
        if self.embeddings_to_use == "first_last_concat":
            return self.task_specific_hidden_size * 2
        if self.embeddings_to_use == "last_only_plus_count":
            return self.task_specific_hidden_size + 1
        return self.task_specific_hidden_size

    def forward(self, vectors):
        """Return fused embeddings of shape (batch, embedding_dim).

        ``vectors`` holds one (batch, seq_len, input_dim) tensor per task.
        """
        if not vectors:
            raise ValueError("Expected at least one task tensor when computing embeddings.")

        batch_size = vectors[0].shape[0]
        task_embeddings = []
        task_embedding_dim = self._get_task_embedding_dim()

        for i, vector in enumerate(vectors):
            if vector is None:
                # A task passed as ``None`` contributes an all-zero embedding.
                zero_output = torch.zeros(
                    batch_size,
                    task_embedding_dim,
                    device=COMPUTE_DEVICE,
                )
                task_embeddings.append(zero_output)
                continue

            vector = vector.to(COMPUTE_DEVICE)
            output = self.models[i].get_embedding(vector)
            # mask denotes timesteps where at least one feature is non-pad so averages ignore padded suffix.
            mask = (vector != PAD_VALUE).any(dim=-1)
            valid_counts = mask.sum(dim=1)
            has_valid = valid_counts.unsqueeze(-1) > 0
            batch_indices = torch.arange(batch_size, device=output.device)

            if self.embeddings_to_use == "last_only":
                # pick the last non-pad timestep; if none exist, fall back to zeros
                last_indices = torch.clamp(valid_counts - 1, min=0)
                last_tokens = output[batch_indices, last_indices, :]
                embedding_to_use = torch.where(
                    has_valid,
                    last_tokens,
                    torch.zeros_like(last_tokens),
                )
            elif self.embeddings_to_use == "average_all":
                embedding_to_use = output.mean(dim=1)
            elif self.embeddings_to_use == "masked_average_all":
                mask_float = mask.unsqueeze(-1).float()
                valid_counts = mask_float.sum(dim=1).clamp_min(1.0)
                masked_sum = (output * mask_float).sum(dim=1)
                embedding_to_use = torch.where(
                    valid_counts > 0,
                    masked_sum / valid_counts,
                    torch.zeros_like(masked_sum),
                )
            elif self.embeddings_to_use == "average_last_10":
                embedding_to_use = output[:, -10:, :].mean(dim=1)
            elif self.embeddings_to_use == "average_last_20":
                embedding_to_use = output[:, -20:, :].mean(dim=1)
            elif self.embeddings_to_use == "average_last_30":
                embedding_to_use = output[:, -30:, :].mean(dim=1)
            elif self.embeddings_to_use == "first_only":
                first_indices = mask.float().argmax(dim=1)
                first_tokens = output[batch_indices, first_indices, :]
                embedding_to_use = torch.where(
                    has_valid,
                    first_tokens,
                    torch.zeros_like(first_tokens),
                )
            elif self.embeddings_to_use == "first_last_concat":
                first_indices = mask.float().argmax(dim=1)
                last_indices = torch.clamp(valid_counts - 1, min=0)
                first_tokens = output[batch_indices, first_indices, :]
                last_tokens = output[batch_indices, last_indices, :]
                first_tokens = torch.where(
                    has_valid,
                    first_tokens,
                    torch.zeros_like(first_tokens),
                )
                last_tokens = torch.where(
                    has_valid,
                    last_tokens,
                    torch.zeros_like(last_tokens),
                )
                embedding_to_use = torch.cat([first_tokens, last_tokens], dim=-1)
            elif self.embeddings_to_use == "last_only_plus_count":
                last_indices = torch.clamp(valid_counts - 1, min=0)
                last_tokens = output[batch_indices, last_indices, :]
                last_tokens = torch.where(
                    has_valid,
                    last_tokens,
                    torch.zeros_like(last_tokens),
                )
                seq_len = max(mask.shape[1], 1)
                count_feature = (valid_counts.to(output.dtype) / float(seq_len)).unsqueeze(-1)
                embedding_to_use = torch.cat([last_tokens, count_feature], dim=-1)
            else:
                raise ValueError(f"Invalid embeddings to use: {self.embeddings_to_use}")

            task_embeddings.append(embedding_to_use)

        fused_output = torch.cat(task_embeddings, dim=-1)
        return fused_output


class TaskFusertoLatentModel(nn.Module):
    """MLP fuser mapping the concatenated task embeddings to the PLVM latent vector."""

    def __init__(self, input_dim, encoder_hidden_dim=64, output_dim=2, dropout_rate=0.0):
        super(TaskFusertoLatentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, encoder_hidden_dim)
        self.fc2 = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        self.fc3 = nn.Linear(encoder_hidden_dim, output_dim)
        self.ln1 = nn.LayerNorm(encoder_hidden_dim)
        self.ln2 = nn.LayerNorm(encoder_hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Map embeddings of shape ``(..., input_dim)`` to latents of shape ``(..., output_dim)``."""
        # Store original shape for reshaping back
        original_shape = x.shape

        # Reshape if needed for layer norm
        if len(original_shape) == 3:
            batch_size, seq_len, features = original_shape
            x = x.reshape(-1, features)

        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        # Reshape back if needed
        if len(original_shape) == 3:
            x = x.reshape(batch_size, seq_len, -1)

        return x


class LatentSpaceDecoder(nn.Module):
    """MLP decoder from the latent vector to regression outputs (softplus if ``positive_only``)."""

    def __init__(
        self,
        input_dim=2,
        hidden_dim=8,
        output_dim=1,
        positive_only=True,
        dropout_rate=0.0,
        num_layers=3,
    ):
        super(LatentSpaceDecoder, self).__init__()
        self.num_layers = num_layers
        self.positive_only = positive_only

        # Linear -> LayerNorm -> ReLU (-> Dropout) blocks followed by a linear output layer.
        layers = []

        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Middle layers: hidden_dim -> hidden_dim
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Last layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.decoder_layers = nn.Sequential(*layers)

    def forward(self, x):
        """Decode latent vectors to regression outputs."""
        x = self.decoder_layers(x)
        if self.positive_only:
            x = F.softplus(x)
        return x


def train_task_specific_model(
    args,
    model,
    dataloader,
    val_loader,
    criterion_loss,
    optimizer,
    scheduler,
    max_norm,
    patience=10,
    validation_interval=1,
    val_baseline_losses: Optional[Dict[str, float]] = None,
    use_amp: bool = True,
):
    """Train a task encoder (sequence-to-sequence targets) and return ``(losses, val_losses)``.

    Validation runs every ``validation_interval`` epochs and training stops early after
    ``patience`` validations without improvement.  ``val_baseline_losses`` (name -> loss)
    are only appended to the per-epoch log for comparison.  ``use_amp`` enables CUDA mixed
    precision.
    """

    losses = []
    val_losses = []
    device = COMPUTE_DEVICE
    model.to(device)
    scaler = GradScaler(enabled=(device.type == "cuda" and use_amp))

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0

        for train_inputs in dataloader:
            optimizer.zero_grad()
            inputs = train_inputs[0].to(device, non_blocking=True)
            labels = train_inputs[1].to(device, non_blocking=True)
            padding_mask = build_padding_mask(inputs)

            with amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(inputs, padding_mask=padding_mask)
                reconstruction_loss = criterion_loss(outputs, labels)
                loss = reconstruction_loss

            scaler.scale(loss).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        # Validation (no need for mixed precision here)
        if epoch % validation_interval == 0:
            val_loss = 0
            with torch.no_grad():
                if val_loader is not None:
                    # Use validation loader if provided
                    for val_inputs in val_loader:
                        inputs = val_inputs[0].to(device, non_blocking=True)
                        labels = val_inputs[1].to(device, non_blocking=True)
                        padding_mask = build_padding_mask(inputs)
                        with amp.autocast(enabled=scaler.is_enabled()):
                            val_output = model(inputs, padding_mask=padding_mask)
                            val_loss += criterion_loss(val_output, labels).item()
                    val_loss = val_loss / len(val_loader)

                val_losses.append(val_loss)
                baseline_suffix = ""
                if val_baseline_losses:
                    baseline_parts = []
                    for baseline_name, baseline_loss in val_baseline_losses.items():
                        if baseline_loss is None:
                            continue
                        try:
                            baseline_parts.append(f"{baseline_name}: {float(baseline_loss):.4f}")
                        except (TypeError, ValueError):
                            baseline_parts.append(f"{baseline_name}: {baseline_loss}")
                    if baseline_parts:
                        baseline_suffix = ", " + ", ".join(baseline_parts)

                logging.info(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {avg_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}{baseline_suffix}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                        return losses, val_losses

    return losses, val_losses


def initialize_model_and_optimizer(args, input_size, output_size=None, num_epochs=1000):
    """Return ``(model, optimizer, scheduler, max_norm)`` for a task encoder.

    ``args.model_type == "lstm"`` builds an :class:`LSTMModel`, anything else a
    :class:`TransformerModel`.  ``output_size`` defaults to ``input_size``.
    """
    hidden_size = args.hidden_size
    if output_size is None:
        output_size = input_size
    num_layers = args.num_layers
    nhead = args.nhead
    dropout = getattr(args, "dropout", 0.1)

    if args.model_type == "lstm":
        model = LSTMModel(
            input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers
        ).to(COMPUTE_DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        scheduler = MultiStepLR(optimizer, milestones=[500, 1500, 2500], gamma=0.1)
        max_norm = 0.1
    else:
        model = TransformerModel(
            input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        ).to(COMPUTE_DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[int(num_epochs * 0.25), int(num_epochs * 0.5), int(num_epochs * 0.75)],
            gamma=0.5,
        )
        max_norm = args.max_norm

    return model, optimizer, scheduler, max_norm


def save_model(model, save_path):
    """Save the model's ``hyperparameters`` and ``state_dict`` to ``save_path``."""
    try:
        # Save the hyperparameters and the state_dict in a single dictionary
        torch.save(
            {
                "hyperparameters": model.hyperparameters,  # Save hyperparameters
                "state_dict": model.state_dict(),  # Save model weights
            },
            save_path,
        )
        print(f"Model saved successfully at: {save_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")


def load_model(model_path, model_class=TransformerModel):
    """Rebuild a model saved by :func:`save_model` and return it in eval mode.

    Returns ``None`` (and prints the error) when the file is missing or cannot be loaded.
    """
    try:
        # Load the saved dictionary
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

        hyperparameters = checkpoint.get("hyperparameters", None)
        state_dict = checkpoint.get("state_dict", None)

        if hyperparameters is None or state_dict is None:
            raise ValueError("Checkpoint is missing hyperparameters or state_dict.")

        # Some checkpoints also record the training ``window_size``, which is not a
        # constructor argument of the model.
        hyperparameters.pop("window_size", None)

        # Initialize the model using the saved hyperparameters
        model = model_class(**hyperparameters)

        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

        # Set the model to evaluation mode
        model.eval()

        return model

    except FileNotFoundError:
        print(f"Model file not found at: {model_path}")
        return None
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None


def compute_loss(
    loss_type,
    model,
    latent_vector,
    y_true,
    latent_dim,
    lambda_l2=0.0001,
    max_kld_loss=1000.0,
    loss_weights=None,
):
    """Return the predictive loss for ``loss_type`` ``"mse"`` or ``"classification"``.

    The L2 penalty is not applied here (``lambda_l2`` is ignored);
    :func:`train_embedding_hierarchical_model` adds it once for all loss types.
    ``max_kld_loss`` is unused.  ``loss_weights`` are per-output weights for the MSE loss.
    """
    if loss_type == "mse":
        return compute_mse_loss(
            model, latent_vector, y_true, latent_dim, lambda_l2, loss_weights=loss_weights
        )
    elif loss_type == "classification":
        return compute_classification_loss(
            model,
            latent_vector,
            y_true,
            lambda_l2=lambda_l2,
            num_classes=getattr(model, "num_classes", 2),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_mse_loss(model, latent_vector, y_true, latent_dim, lambda_l2=0.0001, loss_weights=None):
    """Return the masked MSE of the decoder outputs; NaN targets are ignored.

    With ``loss_weights`` (one per output) the loss is the weighted mean over the non-NaN
    targets.  ``lambda_l2`` is not applied here (see :func:`compute_loss`).
    """
    y_pred = model.decoder(latent_vector)

    # Ignore NaNs in y_true
    mask = ~torch.isnan(y_true)
    diff = (y_pred - y_true) ** 2
    diff = torch.where(mask, diff, torch.zeros_like(diff))

    if loss_weights is not None:
        weights = loss_weights
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        weights = weights.to(diff.device)
        diff = diff * weights
        effective_counts = torch.where(
            mask,
            weights,
            torch.zeros_like(weights),
        ).sum()
    else:
        effective_counts = mask.sum()

    if effective_counts.item() == 0:
        return torch.tensor(0.0, device=latent_vector.device)

    mse_loss = diff.sum() / effective_counts

    return mse_loss


def compute_parameter_l2_penalty(model):
    """
    Return the standard L2 penalty used by the embedding trainer.

    This is the unweighted ``0.5 * sum(param^2)`` term so callers can apply a
    single ``lambda_l2`` factor in one place.
    """
    l2_reg = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        param_penalty = torch.sum(param**2)
        l2_reg = param_penalty if l2_reg is None else l2_reg + param_penalty

    if l2_reg is None:
        return torch.tensor(0.0, device=COMPUTE_DEVICE)
    return l2_reg / 2.0


def compute_classification_loss(model, latent_vector, y_true, lambda_l2=0.0001, num_classes=None):
    """
    Compute cross-entropy loss for classification tasks using decoder logits.

    Shared parameter regularization is applied in ``train_embedding_hierarchical_model``
    so this helper returns the predictive loss only.
    """
    if num_classes is None:
        num_classes = getattr(model, "num_classes", 2)

    outputs = model.decoder(latent_vector)
    logits = outputs[:, :num_classes]

    if y_true.dim() > 1:
        y_true = y_true.view(-1)

    y_true = y_true.long()

    loss = F.cross_entropy(logits, y_true)
    return loss


def compute_grouped_classification_loss(
    model,
    logits,
    y_true,
    sample_ids,
    lambda_l2=0.0001,
    num_classes=None,
):
    """
    Compute cross-entropy after averaging logits within each sample ID.

    Shared parameter regularization is applied in ``train_embedding_hierarchical_model``
    so this helper returns the predictive loss only.
    """
    if num_classes is None:
        num_classes = getattr(model, "num_classes", 2)

    logits = logits[:, :num_classes]
    grouped_logits, grouped_targets, _ = aggregate_classification_logits_by_id(
        logits,
        y_true,
        sample_ids,
    )

    return F.cross_entropy(grouped_logits, grouped_targets)


def _backfill_model_performance_row(existing_row, fieldnames):
    """Map one existing registry row onto ``fieldnames``.

    Missing columns are left empty, except ``dim_reduction_method``, which is filled from
    the row's ``all_arguments_json`` when available.
    """
    migrated_row = {}
    all_arguments = {}
    raw_arguments = existing_row.get("all_arguments_json", "")
    if raw_arguments:
        try:
            all_arguments = json.loads(raw_arguments)
        except Exception:
            all_arguments = {}

    for col in fieldnames:
        value = existing_row.get(col, "")
        if (value == "" or value is None) and col == "dim_reduction_method":
            value = all_arguments.get("dim_reduction_method", "")
        migrated_row[col] = "" if value is None else value
    return migrated_row


def _migrate_model_performance_csv_schema(csv_file, fieldnames):
    """Make sure an existing registry CSV has every column in ``fieldnames``.

    Returns the header to append with: ``fieldnames`` for a new or empty file, otherwise the
    file's own header.  A file whose header lacks some of ``fieldnames`` is rewritten with
    ``fieldnames`` followed by its extra columns, so no stored column is dropped.
    """
    if not os.path.isfile(csv_file) or os.path.getsize(csv_file) == 0:
        return list(fieldnames)

    with open(csv_file, newline="") as file:
        reader = csv.DictReader(file)
        existing_fieldnames = list(reader.fieldnames or [])
        if set(fieldnames).issubset(existing_fieldnames):
            return existing_fieldnames
        merged_fieldnames = list(fieldnames) + [
            col for col in existing_fieldnames if col not in fieldnames
        ]
        rows = [_backfill_model_performance_row(row, merged_fieldnames) for row in reader]

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=merged_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return merged_fieldnames


def update_hierarchical_model_performance_csv(csv_file, args, **kwargs):
    """Append one run's settings and metrics to the run-registry CSV.

    Each column is filled from ``kwargs`` first, then from ``args``, then with a type-based
    default (0, False or empty).  The file is locked while written, so parallel runs can
    share one registry; an existing file keeps its own header.
    """
    # Run-registry columns shared by the PLVM, probe and baseline trainers.  Several of
    # them are only filled by some trainers and stay empty or at their default otherwise.
    fieldnames = [
        "timestamp",
        "model_name",
        "latent_dim",
        "encoder_hidden_size",
        "dropout_rate",
        "lambda_l2",
        "decoder_hidden_dim",
        "decoder_num_layers",
        "seq_length",
        "batch_size",
        "num_epochs",
        "max_trials_per_task",
        "prediction_horizon",
        "num_slices_per_task",
        "learning_rate",
        "model_type",
        "prediction_head_type",
        "loss_type",
        "train_size",
        "val_size",
        "data_completeness_mode",
        "parameters_to_predict",
        "predicted_next_task",
        "naive_train_loss",
        "naive_val_loss",
        "train_loss",
        "val_loss",
        "train_mse",
        "val_mse",
        "naive_train_accuracy",
        "naive_val_accuracy",
        "ml_best_model",
        "ml_best_val_accuracy",
        "ml_svm_train_accuracy",
        "ml_svm_val_accuracy",
        "train_accuracy",
        "val_accuracy",
        "max_norm",
        "tasks",
        "task_specific_model_ids",
        "assumed_distribution_type",
        "embeddings_to_use",
        "input_embeddings_source",
        "feature_mode",
        "input_feature_dim",
        "latent_fusion_type",
        "attention_num_heads",
        "attention_dim",
        "attention_dropout",
        "attention_pool",
        "behavior_label_filter",
        "task_loss_weight_mode",
        "task_loss_weights",
        "multitask_head_mode",
        "use_task_specific_heads",
        "diversity_loss_weight",
        "diversity_loss_type",
        "use_data_augmentation",
        "num_augmented_samples",
        "augmentation_k_neighbors",
        "augmentation_noise_std",
        "augmentation_sampling_mode",
        "pdlvm_log_prob_loss_weight",
        "max_kld_loss",
        "latent_vector_penalty_loss_weight",
        "base_model_path",
        "data_file",
        "run_mode",
        "grad_clip_norm",
        "add_noise",
        "pool_kernel_size",
        "pool_stride",
        "use_early_stopping",
        "fusion_type",
        "dim_reduction_method",
        "fold_number",
        "encoder_dropout_rate",
        "decorder_dropout_rate",
        "heldout_session_ids",
        "all_arguments_json",
    ]
    # Convert args to dict for easier access
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
    else:
        args_dict = dict(args)

    # Prepare the row with robust fallback for missing fields
    row = {}
    for col in fieldnames:
        if col == "timestamp":
            row[col] = get_current_time_str()
        elif col == "tasks":
            val = args_dict.get("tasks", [])
            if isinstance(val, (list, tuple)):
                row[col] = ", ".join(map(str, val))
            else:
                row[col] = str(val)
        elif col == "all_arguments_json":
            # Dump all args and kwargs as JSON
            try:
                all_args = args_dict.copy()
                all_args.update(kwargs)
                row[col] = json.dumps(all_args, default=str)
            except Exception as e:
                row[col] = f"<json error: {e}>"
        elif col in kwargs:
            row[col] = kwargs[col]
        elif col in args_dict:
            row[col] = args_dict[col]
        else:
            # Fallbacks for known types
            if (
                col.endswith("_loss")
                or col.endswith("_weight")
                or col.endswith("_size")
                or col.endswith("_dim")
                or col.endswith("_num")
                or col.endswith("_k_neighbors")
                or col.endswith("_std")
                or col.endswith("_norm")
                or col.endswith("_accuracy")
            ):
                row[col] = 0
            elif col.startswith("use_"):
                row[col] = False
            else:
                row[col] = ""

    import fcntl  # noqa: PLC0415

    lock_file = csv_file + ".lock"
    with open(lock_file, mode="w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            output_fieldnames = _migrate_model_performance_csv_schema(csv_file, fieldnames)
            write_header = not os.path.isfile(csv_file) or os.path.getsize(csv_file) == 0
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=output_fieldnames, restval="")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    logging.info(f"Performance metrics updated in {csv_file}.")


def plot_error_analysis(
    ground_truth, predictions, save_path, title="Error Analysis", parameter_labels=None
):
    """Save error-analysis plots of ``predictions`` against ``ground_truth`` to ``save_path``.

    ``parameter_labels`` gives one display label per output column.
    """
    # Convert to numpy for analysis
    gt_np = ground_truth.cpu().numpy()
    pred_np = predictions.detach().cpu().numpy()
    errors = pred_np - gt_np

    n_params = gt_np.shape[1]

    # Generate parameter labels if not provided
    if parameter_labels is None:
        parameter_labels = [f"Parameter {i + 1}" for i in range(n_params)]

    # Create figure with subplots
    fig, axes = plt.subplots(n_params, 4, figsize=(20, 5 * n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)

    # Calculate error metrics
    mae_per_param = []
    rmse_per_param = []
    bias_per_param = []
    corr_per_param = []

    for i in range(n_params):
        # Filter out NaN values for this parameter
        valid_mask = ~(np.isnan(gt_np[:, i]) | np.isnan(pred_np[:, i]))
        gt_valid = gt_np[valid_mask, i]
        pred_valid = pred_np[valid_mask, i]
        errors_valid = errors[valid_mask, i]

        if len(gt_valid) == 0:
            # Skip this parameter if no valid data
            mae_per_param.append(np.nan)
            rmse_per_param.append(np.nan)
            bias_per_param.append(np.nan)
            corr_per_param.append(np.nan)
            continue

        # Calculate metrics
        mae = np.mean(np.abs(errors_valid))
        rmse = np.sqrt(np.mean(errors_valid**2))
        bias = np.mean(errors_valid)
        corr = np.corrcoef(gt_valid, pred_valid)[0, 1] if len(gt_valid) > 1 else np.nan

        mae_per_param.append(mae)
        rmse_per_param.append(rmse)
        bias_per_param.append(bias)
        corr_per_param.append(corr)

        # 1. Ground Truth vs Predicted
        axes[i, 0].scatter(gt_valid, pred_valid, alpha=0.6, s=20)
        # Add diagonal line
        min_val = min(gt_valid.min(), pred_valid.min())
        max_val = max(gt_valid.max(), pred_valid.max())
        axes[i, 0].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)
        axes[i, 0].set_title(f"{parameter_labels[i]} - GT vs Predicted")
        axes[i, 0].set_xlabel("Ground Truth")
        axes[i, 0].set_ylabel("Predicted")
        axes[i, 0].grid(True, alpha=0.3)

        # Add correlation text
        axes[i, 0].text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=axes[i, 0].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

        # 2. Error Distribution
        axes[i, 1].hist(errors_valid, bins=30, alpha=0.7, edgecolor="black")
        axes[i, 1].axvline(0, color="red", linestyle="--", alpha=0.8)
        axes[i, 1].set_title(f"{parameter_labels[i]} - Error Distribution")
        axes[i, 1].set_xlabel("Error (Predicted - Ground Truth)")
        axes[i, 1].set_ylabel("Frequency")
        axes[i, 1].grid(True, alpha=0.3)

        # Add error statistics
        error_stats_text = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nBias: {bias:.3f}"
        axes[i, 1].text(
            0.05,
            0.95,
            error_stats_text,
            transform=axes[i, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
        )

        # 3. Error vs Ground Truth
        axes[i, 2].scatter(gt_valid, errors_valid, alpha=0.6, s=20)
        axes[i, 2].axhline(0, color="red", linestyle="--", alpha=0.8)
        axes[i, 2].set_title(f"{parameter_labels[i]} - Error vs Ground Truth")
        axes[i, 2].set_xlabel("Ground Truth")
        axes[i, 2].set_ylabel("Error")
        axes[i, 2].grid(True, alpha=0.3)

        # Calculate correlation between error and ground truth
        error_gt_corr = np.corrcoef(gt_valid, errors_valid)[0, 1]
        axes[i, 2].text(
            0.05,
            0.95,
            f"Correlation: {error_gt_corr:.3f}",
            transform=axes[i, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # 4. Error vs Predicted Value
        axes[i, 3].scatter(pred_valid, errors_valid, alpha=0.6, s=20)
        axes[i, 3].axhline(0, color="red", linestyle="--", alpha=0.8)
        axes[i, 3].set_title(f"{parameter_labels[i]} - Error vs Predicted")
        axes[i, 3].set_xlabel("Predicted Value")
        axes[i, 3].set_ylabel("Error")
        axes[i, 3].grid(True, alpha=0.3)

        # Calculate correlation between error and predicted
        error_pred_corr = np.corrcoef(pred_valid, errors_valid)[0, 1]
        axes[i, 3].text(
            0.05,
            0.95,
            f"Correlation: {error_pred_corr:.3f}",
            transform=axes[i, 3].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    plt.tight_layout()
    fig.suptitle(f"{title}", fontsize=12, y=1.02)  # Move title up by setting y > 1
    plt.savefig(save_path, bbox_inches="tight")  # Ensure title is included in saved figure
    plt.close()

    # Return summary statistics for further analysis
    return {
        "mae_per_param": mae_per_param,
        "rmse_per_param": rmse_per_param,
        "bias_per_param": bias_per_param,
        "corr_per_param": corr_per_param,
        "overall_mae": np.mean(mae_per_param),
        "overall_rmse": np.mean(rmse_per_param),
        "overall_bias": np.mean(bias_per_param),
        "overall_corr": np.mean(corr_per_param),
    }


def plot_correlation_heatmap(
    ground_truth,
    predictions,
    save_path,
    title="Parameter Correlation Analysis",
    parameter_labels=None,
):
    """Save correlation heatmaps of parameters and errors to ``save_path``.

    ``parameter_labels`` is not used; the axes are labelled P1, P2, ...
    """
    # Convert to numpy for analysis
    gt_np = ground_truth.cpu().numpy()
    pred_np = predictions.detach().cpu().numpy()
    errors = pred_np - gt_np

    n_params = gt_np.shape[1]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Ground Truth Parameter Correlations
    gt_corr_matrix = np.corrcoef(gt_np.T)
    # Handle case where correlation is a scalar (1x1 matrix)
    if np.isscalar(gt_corr_matrix):
        gt_corr_matrix = np.array([[gt_corr_matrix]])
    im1 = axes[0, 0].imshow(gt_corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 0].set_title("Ground Truth Parameter Correlations")
    axes[0, 0].set_xticks(range(n_params))
    axes[0, 0].set_yticks(range(n_params))
    axes[0, 0].set_xticklabels([f"P{i + 1}" for i in range(n_params)])
    axes[0, 0].set_yticklabels([f"P{i + 1}" for i in range(n_params)])

    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            axes[0, 0].text(
                j,
                i,
                f"{gt_corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Predicted Parameter Correlations
    pred_corr_matrix = np.corrcoef(pred_np.T)
    # Handle case where correlation is a scalar (1x1 matrix)
    if np.isscalar(pred_corr_matrix):
        pred_corr_matrix = np.array([[pred_corr_matrix]])
    im2 = axes[0, 1].imshow(pred_corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 1].set_title("Predicted Parameter Correlations")
    axes[0, 1].set_xticks(range(n_params))
    axes[0, 1].set_yticks(range(n_params))
    axes[0, 1].set_xticklabels([f"P{i + 1}" for i in range(n_params)])
    axes[0, 1].set_yticklabels([f"P{i + 1}" for i in range(n_params)])

    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            axes[0, 1].text(
                j,
                i,
                f"{pred_corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Error Correlations
    error_corr_matrix = np.corrcoef(errors.T)
    # Handle case where correlation is a scalar (1x1 matrix)
    if np.isscalar(error_corr_matrix):
        error_corr_matrix = np.array([[error_corr_matrix]])
    im3 = axes[1, 0].imshow(error_corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 0].set_title("Error Correlations")
    axes[1, 0].set_xticks(range(n_params))
    axes[1, 0].set_yticks(range(n_params))
    axes[1, 0].set_xticklabels([f"P{i + 1}" for i in range(n_params)])
    axes[1, 0].set_yticklabels([f"P{i + 1}" for i in range(n_params)])

    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            axes[1, 0].text(
                j,
                i,
                f"{error_corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    plt.colorbar(im3, ax=axes[1, 0])

    # 4. Error vs Ground Truth Correlations
    error_gt_corr = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            # Filter out NaN values
            valid_mask = ~(np.isnan(gt_np[:, i]) | np.isnan(errors[:, j]))
            if valid_mask.any():
                error_gt_corr[i, j] = np.corrcoef(gt_np[valid_mask, i], errors[valid_mask, j])[0, 1]
            else:
                error_gt_corr[i, j] = np.nan

    # Handle case where correlation is a scalar (1x1 matrix)
    if np.isscalar(error_gt_corr):
        error_gt_corr = np.array([[error_gt_corr]])
    im4 = axes[1, 1].imshow(error_gt_corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_title("Error vs Ground Truth Correlations")
    axes[1, 1].set_xticks(range(n_params))
    axes[1, 1].set_yticks(range(n_params))
    axes[1, 1].set_xticklabels([f"P{i + 1}" for i in range(n_params)])
    axes[1, 1].set_yticklabels([f"P{i + 1}" for i in range(n_params)])

    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            if not np.isnan(error_gt_corr[i, j]):
                axes[1, 1].text(
                    j,
                    i,
                    f"{error_gt_corr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
    plt.colorbar(im4, ax=axes[1, 1])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "gt_corr_matrix": gt_corr_matrix,
        "pred_corr_matrix": pred_corr_matrix,
        "error_corr_matrix": error_corr_matrix,
        "error_gt_corr_matrix": error_gt_corr,
    }


def save_session_specific_plots(
    trained_model,
    multimdalfuser,
    train_x,
    train_y,
    args,
    folder="train",
    id_mapping=None,
    visualize_latent_space=True,
    parameter_labels=None,
    model_name=None,
    color_by_parameters=None,
    color_parameter_labels=None,
    color_parameter_value_labels=None,
):
    """Save diagnostic plots of one split for a trained PLVM head to ``<log_dir>/plots/<folder>``.

    Classification runs get a confusion matrix, per-class probability histograms and, for
    the train split with a 1-D/2-D latent, a latent-space scatter.  Regression runs get
    density, error and correlation plots, a latent-grid plot (train split, latent_dim <= 2)
    and per-player actual-vs-predicted bars (30 sampled players when there are more than 100).

    Args:
    - trained_model: its ``decoder`` is applied to the latent vectors.
    - multimdalfuser: maps ``train_x`` to ``(latent_vector, outputs)``; usually ``trained_model``.
    - id_mapping: row-aligned player IDs used in the sample-plot titles.
    - color_by_parameters, color_parameter_labels, color_parameter_value_labels: optional
      per-session values to color the latent space by, their labels, and readable names of
      discrete values (see :func:`plot_latent_space_for_all_sessions`).
    """
    # Perform the forward pass to get latent vector and parameters
    latent_vector, _ = multimdalfuser(train_x)

    predicted_parameters = trained_model.decoder(
        latent_vector
    )  # Shape: (num_sessions, num_parameters or num_classes)

    # Create plots directory
    plots_dir = f"{args.log_dir}/plots/{folder}"
    os.makedirs(plots_dir, exist_ok=True)

    regression_labels: Optional[List[str]] = None
    class_label_names: Optional[List[str]] = None

    if parameter_labels is not None and len(parameter_labels) > 0:
        regression_labels = list(parameter_labels)

    task_type = str(getattr(args, "task_type", "") or "").strip().lower()
    predicts_classification = task_type == "classification"
    predicts_regression = not predicts_classification
    classification_only = predicts_classification

    classification_logits = None
    predicted_probs = None
    predicted_classes = None
    regression_predictions = None
    classification_targets = None
    regression_targets = None

    if predicts_classification:
        if train_y.dim() == 1:
            classification_targets = train_y
        else:
            classification_targets = train_y[:, 0]
        classification_targets = classification_targets.long()

        num_classes = getattr(trained_model, "num_classes", None)
        if num_classes is None:
            num_classes = int(torch.max(classification_targets).item() + 1)
        classification_logits = predicted_parameters[:, :num_classes]

    if predicts_regression:
        regression_targets = train_y
        regression_predictions = predicted_parameters

    latent_color_parameters = []
    latent_color_labels = []
    latent_color_value_labels = []
    if color_by_parameters is not None and color_parameter_labels is not None:
        latent_color_parameters.extend(list(color_by_parameters))
        latent_color_labels.extend(list(color_parameter_labels))
        if color_parameter_value_labels is not None:
            latent_color_value_labels.extend(list(color_parameter_value_labels))
        else:
            latent_color_value_labels.extend([None] * len(color_parameter_labels))
    elif (color_by_parameters is None) ^ (color_parameter_labels is None):
        logging.warning(
            "color_by_parameters and color_parameter_labels should both be provided; ignoring custom colors."
        )

    if predicts_classification and classification_logits is not None:
        predicted_probs = torch.softmax(classification_logits, dim=1).detach().cpu().numpy()
        true_labels = classification_targets.view(-1).detach().cpu().numpy()
        configured_class_names = list(getattr(args, "class_names", []) or [])
        if len(configured_class_names) == predicted_probs.shape[1]:
            class_label_names = configured_class_names
        else:
            class_label_names = _resolve_class_names(predicted_probs.shape[1])

        classification_dir = os.path.join(plots_dir, "classification")
        title_prefix = (
            f"{folder.capitalize()} Set for {model_name}" if model_name else folder.capitalize()
        )
        generate_classification_visualizations(
            true_labels=true_labels,
            predicted_probs=predicted_probs,
            output_dir=classification_dir,
            class_names=class_label_names,
            title_prefix=title_prefix,
        )

        predicted_classes = np.argmax(predicted_probs, axis=1)
        latent_color_parameters.extend([true_labels, predicted_classes])
        latent_color_labels.extend(["True Class", "Predicted Class"])
        class_label_map = {idx: class_label_names[idx] for idx in range(len(class_label_names))}
        latent_color_value_labels.extend([class_label_map, class_label_map.copy()])

        logging.info("Saved classification-specific visualizations to %s", classification_dir)
    if predicts_regression and regression_targets is not None:
        density_labels = (
            regression_labels[: regression_targets.shape[1]] if regression_labels else None
        )
        predictions_dir = os.path.join(plots_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # Plot parameter densities for both ground truth and predictions
        os.makedirs(f"{plots_dir}/parameter_densities", exist_ok=True)

        plot_parameter_densities(
            ground_truth=regression_targets,
            predictions=regression_predictions,
            save_path=f"{plots_dir}/parameter_densities/parameter_densities.pdf",
            title=f"Parameter Densities - {folder.capitalize()} Set for {model_name}",
            parameter_labels=density_labels,
            unit_labels=["Value", "Density"],
        )

        # Add comprehensive error analysis
        os.makedirs(f"{plots_dir}/error_analysis", exist_ok=True)
        error_stats = plot_error_analysis(
            ground_truth=regression_targets,
            predictions=regression_predictions,
            save_path=f"{plots_dir}/error_analysis/error_analysis.pdf",
            title=f"Error Analysis - {folder.capitalize()} Set for {model_name}",
            parameter_labels=density_labels,
        )

        # Add correlation heatmap analysis
        plot_correlation_heatmap(
            ground_truth=regression_targets,
            predictions=regression_predictions,
            save_path=f"{plots_dir}/error_analysis/correlation_heatmap.pdf",
            title=f"Parameter Correlation Analysis - {folder.capitalize()} Set for {model_name}",
            parameter_labels=density_labels,
        )

        # Log error statistics
        logging.info(f"Error Analysis for {folder} set for {model_name}:")
        logging.info(f"  Overall MAE: {error_stats['overall_mae']:.4f}")
        logging.info(f"  Overall RMSE: {error_stats['overall_rmse']:.4f}")
        logging.info(f"  Overall Bias: {error_stats['overall_bias']:.4f}")
        logging.info(f"  Overall Correlation: {error_stats['overall_corr']:.4f}")
        logging.info(f"  Per-parameter MAE: {error_stats['mae_per_param']}")
        logging.info(f"  Per-parameter RMSE: {error_stats['rmse_per_param']}")

    # Save the latent space plot for all sessions.
    # Dense parameter-surface plots require regression targets, but pure
    # classification runs can still visualize the learned 1D/2D latent vectors.
    save_dir = os.path.join(folder, "..")
    should_plot_latent = folder == "train" and visualize_latent_space
    can_build_dense_latent_grid = (
        should_plot_latent and regression_targets is not None and latent_vector.shape[1] <= 2
    )
    can_plot_classification_latent = (
        should_plot_latent
        and classification_only
        and classification_targets is not None
        and latent_vector.shape[1] <= 2
        and latent_color_parameters
        and latent_color_labels
    )
    if should_plot_latent and latent_vector.shape[1] > 2:
        logging.warning(
            "Skipping dense latent-space plot for model %s in %s split because latent_dim=%d requires an intractable 100^d grid.",
            model_name if model_name else getattr(args, "model_name", "unknown"),
            folder,
            int(latent_vector.shape[1]),
        )

    grid = None
    grid_parameters = None
    if can_build_dense_latent_grid:
        min_vals = latent_vector.detach().min(dim=0)[0].cpu().numpy()
        max_vals = latent_vector.detach().max(dim=0)[0].cpu().numpy()
        points = [
            torch.linspace(min_vals[i], max_vals[i], 100).to(COMPUTE_DEVICE)
            for i in range(latent_vector.shape[1])
        ]
        grid = (
            torch.stack(torch.meshgrid(*points, indexing="ij"), dim=-1)
            .view(-1, latent_vector.shape[1])
            .to(COMPUTE_DEVICE)
        )

        trained_model.eval()  # set the model to evaluation mode to avoid batch norm and dropout

        # Predict parameters for the grid points only when we will actually plot them.
        with torch.no_grad():
            grid_output = trained_model.decoder(grid)
            if predicts_regression and regression_targets is not None:
                grid_parameters = F.softplus(grid_output).detach().cpu().numpy()

    if can_build_dense_latent_grid and grid is not None and grid_parameters is not None:
        plot_latent_space_for_all_sessions(
            grid.cpu().numpy(),
            grid_parameters,
            latent_vector.detach().cpu().numpy(),
            regression_predictions.detach().cpu().numpy(),
            args.tasks,
            args.log_dir,
            save_dir,
            parameter_labels=density_labels,
        )

    if (
        can_build_dense_latent_grid
        and grid is not None
        and grid_parameters is not None
        and latent_color_parameters
        and latent_color_labels
    ):
        plot_latent_space_for_all_sessions(
            grid.cpu().numpy(),
            grid_parameters,
            latent_vector.detach().cpu().numpy(),
            regression_predictions.detach().cpu().numpy(),
            args.tasks,
            args.log_dir,
            save_dir,
            parameter_labels=density_labels,
            color_by_parameters=latent_color_parameters,
            color_parameter_labels=latent_color_labels,
            model_name=model_name,
            color_parameter_value_labels=latent_color_value_labels
            if latent_color_value_labels
            else None,
        )

    if can_plot_classification_latent:
        latent_vectors_np = latent_vector.detach().cpu().numpy()
        plot_latent_space_for_all_sessions(
            latent_vectors_np,
            np.empty((latent_vectors_np.shape[0], 0), dtype=np.float32),
            latent_vectors_np,
            np.empty((latent_vectors_np.shape[0], 0), dtype=np.float32),
            args.tasks,
            args.log_dir,
            save_dir,
            parameter_labels=None,
            color_by_parameters=latent_color_parameters,
            color_parameter_labels=latent_color_labels,
            model_name=model_name,
            color_parameter_value_labels=latent_color_value_labels
            if latent_color_value_labels
            else None,
        )
        _save_classification_latent_plot_data(
            log_dir=args.log_dir,
            split_name=folder,
            latent_vectors=latent_vectors_np,
            true_labels=classification_targets.view(-1).detach().cpu().numpy(),
            predicted_labels=np.asarray(predicted_classes),
            predicted_probs=np.asarray(predicted_probs),
            class_names=class_label_names,
            model_name=model_name,
        )
        logging.info(
            "Saved classification latent-space visualization to %s/plots/%s",
            args.log_dir,
            save_dir,
        )

    if not predicts_regression or regression_targets is None:
        return

    # Actual-vs-predicted bar plots for a sample of players.
    num_outputs = int(regression_targets.shape[1]) if regression_targets.dim() > 1 else 1
    if num_outputs <= 0:
        return

    if regression_labels:
        output_names = list(regression_labels[:num_outputs])
    else:
        output_names = list(getattr(args, "predicted_tasks", []) or [])
        if not output_names:
            output_names = list(getattr(args, "tasks", []) or [])
        if len(output_names) < num_outputs:
            output_names.extend([f"task_{idx}" for idx in range(len(output_names), num_outputs)])
        output_names = output_names[:num_outputs]

    num_sessions = regression_targets.shape[0]
    sampled_sessions = (
        range(num_sessions) if num_sessions <= 100 else random.sample(range(num_sessions), 30)
    )
    for session in sampled_sessions:
        fig, axs = plt.subplots(1, num_outputs, figsize=(5 * num_outputs, 4))
        if num_outputs == 1:
            axs = [axs]
        else:
            axs = np.asarray(axs).flatten()

        session_y_data = regression_targets[session]
        session_predictions = regression_predictions[session]
        if session_y_data.dim() == 0:
            session_y_data = session_y_data.unsqueeze(0)
        if session_predictions.dim() == 0:
            session_predictions = session_predictions.unsqueeze(0)

        for output_idx, output_name in enumerate(output_names):
            ax = axs[output_idx]
            true_val = session_y_data[output_idx].cpu().item()
            pred_val = session_predictions[output_idx].item()
            ax.bar([0, 1], [true_val, pred_val], color=["#1f77b4", "#ff7f0e"], width=0.6)
            ax.set_title(output_name)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Actual", "Predicted"])
            ax.set_ylabel("Target value")
            ax.text(
                0.5,
                0.9,
                f"MSE={(true_val - pred_val) ** 2:.2f}",
                ha="center",
                transform=ax.transAxes,
                fontsize=10,
            )
        if id_mapping is not None:
            title = f"Player ID: {id_mapping[session]}"
            idx_label = id_mapping[session]
        else:
            title = f"Session ID: {session}"
            idx_label = session
        plt.suptitle(title)
        plt.tight_layout()
        os.makedirs(f"{args.log_dir}/plots/{folder}/samples", exist_ok=True)
        plt.savefig(f"{args.log_dir}/plots/{folder}/samples/sess_{idx_label}_{session}.png")
        plt.close()


def _prepare_latent_space_data(grid_points, train_latent_vectors):
    """Reduce the latent grid and training latents to 2-D for plotting.

    Returns ``(grid_2d, train_2d, grid_sample_indices)``.  Latents with more than two
    dimensions are reduced with UMAP and the grid is subsampled to 20,000 points;
    ``grid_sample_indices`` are the kept grid rows so callers can subsample per-grid-point
    values the same way.  Otherwise the inputs are returned unchanged with ``None``.
    """
    if grid_points.shape[1] > 2:
        # Initialize UMAP with parameters tuned for visualization
        umap_reducer = UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42
        )

        # Fit and transform the grid points
        latent_2d_grid = umap_reducer.fit_transform(grid_points)

        # Transform the training vectors using the fitted UMAP
        trained_predicted_vector = umap_reducer.transform(train_latent_vectors)

        # Randomly sample 20K data points to cover the entire space
        sample_indices = np.random.choice(latent_2d_grid.shape[0], 20000, replace=False)
        latent_2d_grid = latent_2d_grid[sample_indices]

        return latent_2d_grid, trained_predicted_vector, sample_indices
    else:
        return grid_points, train_latent_vectors, None


def _create_latent_space_figure(num_plots, max_cols=3, figsize_per_plot=(8, 8)):
    """Return ``(fig, axs, num_cols, num_rows)`` with a flat list of ``num_plots`` axes."""
    num_cols = min(max_cols, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig_width = figsize_per_plot[0] * num_cols
    fig_height = figsize_per_plot[1] * num_rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    # Handle single subplot case
    if num_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    return fig, axs, num_cols, num_rows


def calculate_latent_parameter_rmse(x_values, y_values, color_values):
    """Return an annotation with the 5-fold CV RMSE of a random forest predicting ``color_values``.

    RMSE is reported for each latent dimension alone and for both together.  Returns an
    empty string for fewer than 10 points and "RMSE calculation failed" if the fit raises.
    """
    try:
        # Prepare data for Random Forest
        X = np.column_stack([x_values, y_values])  # Features: both latent dimensions
        y = color_values  # Target: parameter values

        # Check if we have enough data for cross-validation
        if len(X) >= 10:  # Need at least 10 samples for meaningful CV
            # Calculate RMSE for individual dimensions
            # Dimension 1 only
            rf_dim1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rmse_dim1 = np.sqrt(
                -cross_val_score(
                    rf_dim1, x_values.reshape(-1, 1), y, scoring="neg_mean_squared_error", cv=5
                )
            ).mean()

            # Dimension 2 only
            rf_dim2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rmse_dim2 = np.sqrt(
                -cross_val_score(
                    rf_dim2, y_values.reshape(-1, 1), y, scoring="neg_mean_squared_error", cv=5
                )
            ).mean()

            # Both dimensions together
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rmse_overall = np.sqrt(
                -cross_val_score(rf_model, X, y, scoring="neg_mean_squared_error", cv=5)
            ).mean()

            # Return RMSE information
            return f"\nRandom Forest [ RMSE(Dim1): {rmse_dim1:.3f}, RMSE(Dim2): {rmse_dim2:.3f}, Overall: {rmse_overall:.3f}]"
        else:
            # Too few points for 5-fold cross-validation; omit the annotation.
            return ""

    except Exception as e:
        # The RMSE annotation is optional; log the failure and keep plotting.
        logging.error(f"RMSE calculation failed: {e}")
        logging.error(traceback.format_exc())
        return "RMSE calculation failed"


def _plot_latent_scatter(
    ax,
    x_values,
    y_values,
    color_values,
    title,
    colorbar_label,
    cmap="viridis",
    alpha=0.7,
    s=30,
    show_correlation=True,
    discrete_values=None,
    discrete_value_labels=None,
):
    """Draw a latent-space scatter colored by ``color_values`` with a colorbar or legend.

    ``show_correlation`` appends the :func:`calculate_latent_parameter_rmse` annotation to
    the title.  ``discrete_values`` switches to categorical coloring with a legend, labelled
    through ``discrete_value_labels`` (dict or list) when given.
    """
    has_discrete = discrete_values is not None and len(discrete_values) > 0
    if has_discrete:
        sorted_vals = np.array(sorted(discrete_values))
        value_to_color = {}
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not default_colors:
            default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for idx, val in enumerate(sorted_vals):
            value_to_color[val] = default_colors[idx % len(default_colors)]

        color_values = np.asarray(color_values)
        mask = np.isin(color_values, sorted_vals)
        if not np.all(mask):
            logging.warning(
                "Some color values were not mapped to discrete categories and will be skipped"
            )
        x_plot = x_values[mask]
        y_plot = y_values[mask]
        plot_colors = [value_to_color[val] for val in color_values[mask]]
        scatter = ax.scatter(x_plot, y_plot, c=plot_colors, s=s, alpha=alpha)
    else:
        scatter = ax.scatter(x_values, y_values, c=color_values, cmap=cmap, s=s, alpha=alpha)

    # Add colorbar
    if has_discrete:

        def _resolve_label(val):
            if discrete_value_labels is None:
                return str(val)
            if isinstance(discrete_value_labels, dict):
                return str(discrete_value_labels.get(val, val))
            if isinstance(discrete_value_labels, (list, tuple)):
                idx = int(val)
                if 0 <= idx < len(discrete_value_labels):
                    return str(discrete_value_labels[idx])
            return str(val)

        legend_handles = []
        for val in sorted_vals:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=value_to_color[val],
                    markersize=6,
                    label=_resolve_label(val),
                )
            )
        ax.legend(handles=legend_handles, title=colorbar_label, loc="best")
    else:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(colorbar_label)

    # Calculate and display RMSE from Random Forest prediction if requested
    if show_correlation and len(np.unique(y_values)) > 1:  # Only for 2D plots
        rmse_info = calculate_latent_parameter_rmse(x_values, y_values, color_values)
        full_title = f"{title} {rmse_info}"
    else:
        full_title = title

    # Set title and labels
    ax.set_title(full_title, fontsize=10)
    ax.set_xlabel("Latent Dimension 1")
    if len(np.unique(y_values)) > 1:  # Only add y-label if not all y-values are the same
        ax.set_ylabel("Latent Dimension 2")

    # Add grid
    ax.grid(True, alpha=0.3)

    return scatter


def _save_latent_space_plot(fig, log_dir, folder, filename, title=None):
    """Save ``fig`` as ``<log_dir>/plots/<folder>/<filename>``, with an optional overall title."""
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Create directory and save
    os.makedirs(f"{log_dir}/plots/{folder}", exist_ok=True)
    plt.savefig(f"{log_dir}/plots/{folder}/{filename}")
    plt.close()


def _save_classification_latent_plot_data(
    log_dir: str,
    split_name: str,
    latent_vectors: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_probs: np.ndarray,
    class_names: List[str],
    model_name: Optional[str] = None,
) -> None:
    """Persist the exact coordinates and labels used for classification latent plots."""
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if latent_vectors.ndim != 2 or latent_vectors.shape[1] < 1:
        raise ValueError("latent_vectors must have shape [num_samples, latent_dim>=1].")
    if predicted_probs.ndim != 2:
        raise ValueError("predicted_probs must have shape [num_samples, num_classes].")

    split_name = str(split_name or "train").strip().lower()
    latent_x = latent_vectors[:, 0]
    latent_y = (
        latent_vectors[:, 1]
        if latent_vectors.shape[1] > 1
        else np.zeros(latent_vectors.shape[0], dtype=latent_vectors.dtype)
    )
    prob_columns = [
        "predicted_prob_{}".format(class_idx) for class_idx in range(predicted_probs.shape[1])
    ]
    csv_path = os.path.join(
        plots_dir,
        "{}_classification_latent_space_plot_data.csv".format(split_name),
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "split",
                "model_name",
                "latent_x",
                "latent_y",
                "true_class_id",
                "true_class_label",
                "predicted_class_id",
                "predicted_class_label",
                *prob_columns,
            ],
        )
        writer.writeheader()
        for sample_index in range(latent_vectors.shape[0]):
            true_class_id = int(true_labels[sample_index])
            predicted_class_id = int(predicted_labels[sample_index])
            row = {
                "sample_index": sample_index,
                "split": split_name,
                "model_name": model_name or "",
                "latent_x": float(latent_x[sample_index]),
                "latent_y": float(latent_y[sample_index]),
                "true_class_id": true_class_id,
                "true_class_label": class_names[true_class_id],
                "predicted_class_id": predicted_class_id,
                "predicted_class_label": class_names[predicted_class_id],
            }
            for class_idx, column_name in enumerate(prob_columns):
                row[column_name] = float(predicted_probs[sample_index, class_idx])
            writer.writerow(row)

    metadata_path = os.path.join(
        plots_dir,
        "{}_classification_latent_space_plot_metadata.json".format(split_name),
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": split_name,
                "model_name": model_name or "",
                "latent_dim": int(latent_vectors.shape[1]),
                "plot_filename": "latent_space_colored_by_parameters.pdf",
                "class_names": list(class_names),
                "probability_columns": prob_columns,
                "used_umap": False,
            },
            handle,
            indent=2,
        )

    logging.info(
        "Saved classification latent-plot data to %s and %s",
        csv_path,
        metadata_path,
    )


def plot_latent_space_for_all_sessions(
    grid_points,
    grid_preds,
    train_latent_vectors,
    trained_predicted_parameters,
    task_names,
    log_dir,
    folder,
    parameter_labels=None,
    color_by_parameters=None,
    color_parameter_labels=None,
    model_name=None,
    color_parameter_value_labels=None,
):
    """Plot the latent space of all sessions in one of two modes.

    - Custom-color mode (``color_by_parameters`` and ``color_parameter_labels`` given): one
      scatter of the session latents per coloring, saved as
      ``latent_space_colored_by_parameters.pdf``.  ``color_parameter_value_labels`` gives
      readable names of discrete values.
    - Grid mode: one panel per decoder output, coloring the grid points by the predicted
      output (``grid_preds``) and overlaying the session latents, saved as
      ``latent_space.pdf``.  Panels are titled with ``parameter_labels`` or "Output 1", ...

    ``trained_predicted_parameters`` and ``task_names`` are not used.
    """
    latent_2d_grid, trained_predicted_vector, grid_sample_indices = _prepare_latent_space_data(
        grid_points, train_latent_vectors
    )

    # Keep the grid predictions aligned with the subsampled UMAP grid.
    if grid_sample_indices is not None:
        grid_preds = grid_preds[grid_sample_indices]

    # Determine plotting mode
    if color_by_parameters is not None and color_parameter_labels is not None:
        # Custom parameter mode
        if len(color_by_parameters) != len(color_parameter_labels):
            raise ValueError(
                "color_by_parameters and color_parameter_labels must have the same length"
            )
        if color_parameter_value_labels is not None and len(color_parameter_value_labels) not in (
            0,
            len(color_parameter_labels),
        ):
            raise ValueError(
                "color_parameter_value_labels must be empty or match length of color_parameter_labels"
            )

        if len(color_by_parameters) == 0:
            return

        num_plots = len(color_by_parameters)
        fig, axs, num_cols, num_rows = _create_latent_space_figure(num_plots, max_cols=3)

        # Plot each custom parameter
        for param_idx, (param_values, param_label) in enumerate(
            zip(color_by_parameters, color_parameter_labels)
        ):
            # Ensure param_values has the right shape
            if len(param_values) != len(train_latent_vectors):
                raise ValueError(
                    f"Parameter {param_label} has {len(param_values)} values but expected {len(train_latent_vectors)}"
                )
            value_label_map = None
            if color_parameter_value_labels is not None and len(color_parameter_value_labels) > 0:
                value_label_map = color_parameter_value_labels[param_idx]

            param_values = np.asarray(param_values)
            unique_values = np.unique(param_values)
            try:
                is_discrete = np.issubdtype(param_values.dtype, np.integer) or np.issubdtype(
                    param_values.dtype, np.bool_
                )
            except TypeError:
                is_discrete = False
            discrete_ticks = unique_values if is_discrete else None
            if is_discrete:
                if len(unique_values) <= 10:
                    cmap = "tab10"
                elif len(unique_values) <= 20:
                    cmap = "tab20"
                else:
                    cmap = "gist_ncar"
            else:
                cmap = "viridis"

            # Determine x and y values
            if latent_2d_grid.shape[1] == 1:
                x_values = trained_predicted_vector[:, 0]
                y_values = np.zeros_like(x_values)
            else:
                x_values = trained_predicted_vector[:, 0]
                y_values = trained_predicted_vector[:, 1]

            # Create scatter plot using modular component
            _plot_latent_scatter(
                axs[param_idx],
                x_values,
                y_values,
                param_values,
                f"Latent Space Colored by {param_label}",
                param_label,
                cmap=cmap,
                show_correlation=not is_discrete,
                discrete_values=discrete_ticks,
                discrete_value_labels=value_label_map,
            )

        # Hide unused subplots
        for idx in range(num_plots, len(axs)):
            axs[idx].set_visible(False)

        # Save using modular component
        title = (
            f"Latent Space Visualization by Parameters - {train_latent_vectors.shape[0]} sessions"
        )
        if model_name:
            title += f" - {model_name}"
        _save_latent_space_plot(
            fig, log_dir, folder, "latent_space_colored_by_parameters.pdf", title
        )

    else:
        if parameter_labels:
            resolved_labels = list(parameter_labels)
        else:
            resolved_labels = [f"Output {idx + 1}" for idx in range(grid_preds.shape[1])]

        if resolved_labels:
            num_plots = len(resolved_labels)
            fig, axs, num_cols, num_rows = _create_latent_space_figure(
                num_plots, max_cols=min(3, num_plots)
            )

            for idx, label in enumerate(resolved_labels):
                if idx >= grid_preds.shape[1]:
                    logging.warning(f"Skipping parameter {label} as index {idx} is out of bounds")
                    continue

                values = grid_preds[:, idx].flatten()
                if latent_2d_grid.shape[1] == 1:
                    x_values = latent_2d_grid[:, 0]
                    y_values = np.zeros_like(x_values)
                else:
                    x_values = latent_2d_grid[:, 0]
                    y_values = latent_2d_grid[:, 1]

                _plot_latent_scatter(
                    axs[idx],
                    x_values,
                    y_values,
                    values,
                    f"Latent Space Colored by {label}",
                    label,
                    cmap="viridis",
                    show_correlation=True,
                )

                axs[idx].scatter(
                    trained_predicted_vector[:, 0],
                    trained_predicted_vector[:, 1]
                    if latent_2d_grid.shape[1] > 1
                    else np.zeros_like(trained_predicted_vector[:, 0]),
                    s=20,
                    alpha=0.8 if latent_2d_grid.shape[1] > 1 else 0.4,
                    marker="x" if latent_2d_grid.shape[1] > 1 else "|",
                    color="red",
                )

            for idx in range(len(resolved_labels), len(axs)):
                axs[idx].set_visible(False)

            title = f"Latent Space Colored by Parameters - {train_latent_vectors.shape[0]} sessions"
            if model_name:
                title += f" - {model_name}"
            _save_latent_space_plot(fig, log_dir, folder, "latent_space.pdf", title)


def compute_naive_baseline_losses(
    train_y, val_y, train_item_level_data, val_item_level_data, metric="mse", args=None
):
    """Train/validation loss of a constant predictor fitted on train.

    MSE of the train mean for regression, or cross-entropy of the train class-frequency prior
    for ``metric="classification"``. Also returns the train mean (or the majority class).
    """
    if metric == "classification":
        train_flat = train_y.view(-1).long()
        val_flat = val_y.view(-1).long()

        # Preserve the majority-class label for callers that report a simple
        # accuracy baseline, but compute the naive loss in the same
        # cross-entropy space used by the trained classifier.
        mode_result = torch.mode(train_flat)
        mean_train_y = mode_result.values

        num_classes = int(getattr(args, "num_classes", 0) or 0)
        if train_flat.numel() > 0:
            num_classes = max(num_classes, int(train_flat.max().item()) + 1)
        if val_flat.numel() > 0:
            num_classes = max(num_classes, int(val_flat.max().item()) + 1)
        if num_classes < 1:
            raise ValueError("Classification naive baseline requires at least one class.")

        class_counts = torch.bincount(
            train_flat,
            minlength=num_classes,
        ).to(device=train_flat.device, dtype=torch.float32)
        class_probs = class_counts / class_counts.sum().clamp_min(1.0)
        # Keep the baseline loss finite if validation contains a class unseen
        # in the training split.
        class_probs = torch.clamp(class_probs, min=1e-7)
        class_probs = class_probs / class_probs.sum()
        naive_logits = torch.log(class_probs)

        train_logits = naive_logits.unsqueeze(0).expand(train_flat.shape[0], -1)
        val_logits = naive_logits.unsqueeze(0).expand(val_flat.shape[0], -1)
        naive_train_loss = F.cross_entropy(train_logits, train_flat)
        naive_val_loss = F.cross_entropy(val_logits, val_flat)

        return naive_train_loss, naive_val_loss, mean_train_y

    # Calculate the mean of the training data
    mean_train_y = torch.nanmean(train_y, dim=0)  # Shape: (num_tasks * num_parameters)

    if metric == "mse":
        # Compute the square difference for training data
        train_square_diff = (mean_train_y.expand_as(train_y) - train_y) ** 2

        # Remove NaN values for training data
        train_valid_mask = ~torch.isnan(train_square_diff)
        train_square_diff = train_square_diff[train_valid_mask]

        # Compute the mean of the valid square differences for training data
        naive_train_loss = train_square_diff.mean()

        # Compute the square difference for validation data
        val_square_diff = (mean_train_y.expand_as(val_y) - val_y) ** 2

        # Remove NaN values for validation data
        val_valid_mask = ~torch.isnan(val_square_diff)
        val_square_diff = val_square_diff[val_valid_mask]

        # Compute the mean of the valid square differences for validation data
        naive_val_loss = val_square_diff.mean()

    else:
        raise ValueError(f"Metric {metric} not supported")

    return naive_train_loss, naive_val_loss, mean_train_y


def _ensure_tensor_labels(labels):
    if torch.is_tensor(labels):
        return labels.detach().long()
    return torch.tensor(labels, dtype=torch.long)


def compute_classification_metrics(y_true, y_pred, num_classes=None):
    """
    Compute accuracy, macro F1, per-class F1, and confusion matrix.
    Accepts tensors or numpy arrays for inputs.
    """
    if torch.is_tensor(y_pred) and y_pred.dim() > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    y_true_tensor = _ensure_tensor_labels(y_true).view(-1)
    y_pred_tensor = _ensure_tensor_labels(y_pred).view(-1)

    y_true_np = y_true_tensor.cpu().numpy()
    y_pred_np = y_pred_tensor.cpu().numpy()

    if num_classes is None and y_true_np.size:
        num_classes = int(max(y_true_np.max(), y_pred_np.max()) + 1)
    elif num_classes is None:
        num_classes = 1

    accuracy = accuracy_score(y_true_np, y_pred_np) if y_true_np.size else 0.0
    macro_f1 = (
        f1_score(y_true_np, y_pred_np, average="macro", zero_division=0) if y_true_np.size else 0.0
    )

    per_class_f1 = {}
    for class_idx in range(num_classes):
        per_class_f1[class_idx] = f1_score(
            y_true_np,
            y_pred_np,
            labels=[class_idx],
            average="macro",
            zero_division=0,
        )

    cm = confusion_matrix(
        y_true_np,
        y_pred_np,
        labels=list(range(num_classes)),
    ).astype(int)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "f1_per_class": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "support": int(y_true_np.size),
    }


def _resolve_class_names(num_classes: int) -> List[str]:
    """
    Return generic class names ``"Class 0"``, ``"Class 1"``, ...

    Used when the run does not provide ``args.class_names`` of the right length.
    """
    return [f"Class {idx}" for idx in range(num_classes)]


def generate_classification_visualizations(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    output_dir: str,
    class_names: List[str],
    title_prefix: str,
):
    """
    Generate key classification visualisations: confusion matrix and per-class probability histograms.
    """
    os.makedirs(output_dir, exist_ok=True)

    if true_labels.ndim != 1:
        true_labels = true_labels.reshape(-1)

    predicted_labels = np.argmax(predicted_probs, axis=1)
    num_classes = len(class_names)
    labels = np.arange(num_classes)

    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title_prefix} - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.pdf"))
    plt.close(fig)

    # Per-class probability calibration plots
    fig, axes = plt.subplots(num_classes, 1, figsize=(7, 3 * num_classes), sharex=True)
    if num_classes == 1:
        axes = [axes]

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]
        mask = true_labels == class_idx
        if np.any(mask):
            sns.histplot(
                predicted_probs[mask, class_idx],
                bins=20,
                kde=True,
                stat="probability",
                color="steelblue",
                ax=ax,
            )
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"P(pred = {class_name} | true = {class_name})")
        ax.set_ylabel("Density")
    axes[-1].set_xlabel("Predicted probability")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_class_probability_histograms.pdf"))
    plt.close(fig)

    metrics = compute_classification_metrics(
        torch.tensor(true_labels, dtype=torch.long),
        torch.tensor(predicted_labels, dtype=torch.long),
        num_classes=num_classes,
    )
    metrics_path = os.path.join(output_dir, "classification_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def log_task_specific_model_performance(
    X_train_size, X_val_size, train_loss, val_loss, args, log_csv_path
):
    """Append one task-encoder run (sizes, losses, hyperparameters) to the encoder registry CSV."""
    # read csv file if it exists else create it
    if os.path.exists(log_csv_path):
        with open(log_csv_path, mode="r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]
    else:
        data = [
            [
                "datetime",
                "model_name",
                "seq_length",
                "prediction_horizon",
                "batch_size",
                "num_epochs",
                "max_norm",
                "fold_number",
                "learning_rate",
                "model_type",
                "loss_type",
                "run_mode",
                "task_label",
                "hidden_size",
                "output_size",
                "num_layers",
                "nhead",
                "X_train_size",
                "X_val_size",
                "train_loss",
                "val_loss",
            ]
        ]

    # Log model hyperparameters
    hyperparameters = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.model_name,
        args.seq_length,
        args.prediction_horizon,
        args.batch_size,
        args.num_epochs,
        args.max_norm,
        args.fold_number,
        args.learning_rate,
        args.model_type,
        args.loss_type,
        args.run_mode,
        args.task_label,
        args.hidden_size,
        args.output_size,
        args.num_layers,
        args.nhead,
        X_train_size,
        X_val_size,
    ]

    # Append losses to hyperparameters
    hyperparameters.extend([train_loss, val_loss])
    data.append(hyperparameters)

    # Write the updated data back to the csv file
    with open(log_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    logging.info(f"Model performance logged in {log_csv_path}")


def update_task_specific_model_reference_file(args, reference_file_path=None, validation_ids=None):
    """Record ``args.model_name`` as the reference encoder of ``args.task_label``.

    The default file is
    ``<ARTIFACT_ROOT>/models/best_task_model_ids_<horizon>steps_<YYYYmmdd>.json``.
    """
    if reference_file_path is None:
        today_date = datetime.now().strftime("%Y%m%d")
        reference_file_name = (
            f"best_task_model_ids_{args.prediction_horizon}steps_{today_date}.json"
        )
        reference_file_path = os.path.join(str(artifact_paths.MODELS_DIR), reference_file_name)

    # Load or create reference data
    if os.path.exists(reference_file_path):
        with open(reference_file_path, "r") as f:
            reference_data = json.load(f)
    else:
        reference_data = {
            "tasks": {},
            "seq_length": None,
            "fold_number": None,
            "validation_ids": None,
        }

    if args.task_label not in reference_data["tasks"]:
        reference_data["tasks"][args.task_label] = {}

    reference_data["tasks"][args.task_label] = args.model_name
    reference_data["seq_length"] = args.seq_length
    reference_data["fold_number"] = args.fold_number
    reference_data["validation_ids"] = validation_ids

    logging.info(
        f"Updating reference file {reference_file_path} with model name {args.model_name} for task label {args.task_label}"
    )

    # Save updated reference data
    with open(reference_file_path, "w") as f:
        json.dump(reference_data, f, indent=4)

    logging.info(
        f"Reference file {reference_file_path} updated with model name {args.model_name} for task label {args.task_label}"
    )


class ClassificationDecoder(nn.Module):
    """
    Decoder used when the hierarchical model performs classification.
    A stack of Linear -> LayerNorm -> ReLU -> Dropout blocks followed by a final classifier layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout_rate: float = 0.0,
        num_layers: int = 3,
        use_softmax: bool = False,
    ):
        super().__init__()
        self.use_softmax = use_softmax

        layers: List[nn.Module] = []
        in_dim = input_dim
        # Build num_layers - 1 hidden blocks
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        # Final classifier layer (logits)
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for latent vectors."""
        for layer in self.classifier_layers:
            x = layer(x)
        if self.use_softmax:
            x = F.softmax(x, dim=-1)
        return x


class EmbeddingHierarchicalModel(nn.Module):
    """
    PLVM head on pre-computed embeddings: an MLP fuser maps the concatenated task
    embeddings to a low-dimensional latent vector, and a decoder maps the latent vector to
    the prediction.  ``task_type="classification"`` uses a :class:`ClassificationDecoder`
    (logits); ``task_type="regression"`` uses a :class:`LatentSpaceDecoder`.
    """

    def __init__(
        self,
        embedding_dim,
        low_dim_size=2,
        num_output_parameters=6,
        encoder_hidden_size=128,
        decoder_hidden_dim=8,
        decoder_num_layers=4,
        positive_only=True,
        encoder_dropout_rate=0.0,
        decoder_dropout_rate=0.0,
        task_type: str = "regression",
        num_classes: int = 3,
    ):
        super(EmbeddingHierarchicalModel, self).__init__()

        self.task_type = task_type
        self.num_classes = num_classes
        self.produces_latent = True
        self.prediction_head_type = "hierarchical"
        self.embedding_dim = embedding_dim

        self.low_dim_mapper = TaskFusertoLatentModel(
            input_dim=embedding_dim,
            encoder_hidden_dim=encoder_hidden_size,
            output_dim=low_dim_size,
            dropout_rate=encoder_dropout_rate,
        )

        self.classification_decoder: Optional[nn.Module] = None
        self.regression_decoder: Optional[nn.Module] = None

        if task_type == "classification":
            self.classification_decoder = ClassificationDecoder(
                input_dim=low_dim_size,
                hidden_dim=decoder_hidden_dim,
                num_classes=num_classes,
                dropout_rate=decoder_dropout_rate,
                num_layers=decoder_num_layers,
                use_softmax=False,  # keep logits for CrossEntropyLoss
            )

        elif task_type == "regression":
            self.regression_decoder = LatentSpaceDecoder(
                input_dim=low_dim_size,
                output_dim=num_output_parameters,
                hidden_dim=decoder_hidden_dim,
                num_layers=decoder_num_layers,
                positive_only=positive_only,
                dropout_rate=decoder_dropout_rate,
            )
        else:
            raise ValueError(
                "task_type must be 'classification' or 'regression', got {!r}.".format(task_type)
            )

        self.decoder = (
            self.classification_decoder
            if task_type == "classification"
            else self.regression_decoder
        )
        self.hyperparameters = {
            "embedding_dim": embedding_dim,
            "low_dim_size": low_dim_size,
            "num_parameters": num_output_parameters,
            "encoder_hidden_size": encoder_hidden_size,
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_num_layers": decoder_num_layers,
            "positive_only": positive_only,
            "encoder_dropout_rate": encoder_dropout_rate,
            "decoder_dropout_rate": decoder_dropout_rate,
            "task_type": task_type,
            "num_classes": num_classes,
            "prediction_head_type": "hierarchical",
        }

    def forward(self, embeddings):
        """Return ``(latent_vector, decoded_output)``: class logits or regression outputs."""
        latent_vector = self.low_dim_mapper(embeddings)
        decoded_output = self.decoder(latent_vector)
        return latent_vector, decoded_output


class EmbeddingMLPModel(nn.Module):
    """A simple encoder-only MLP that maps embeddings directly to predictions."""

    def __init__(
        self,
        embedding_dim,
        num_parameters=6,
        dropout_rate: float = 0.0,
        positive_only: bool = True,
        task_type: str = "regression",
        num_layers: int = 5,
        hidden_dim: int = 128,
        hidden_sizes=None,
        num_classes: int = 3,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        self.positive_only = positive_only and task_type != "classification"
        self.produces_latent = False
        self.prediction_head_type = "mlp"
        self.embedding_dim = embedding_dim

        layers = []
        in_dim = embedding_dim
        if hidden_sizes is not None:
            if isinstance(hidden_sizes, int):
                layer_dims = [hidden_sizes]
            else:
                layer_dims = list(hidden_sizes)
        else:
            layer_dims = [hidden_dim] * max(num_layers - 1, 0)

        for layer_hidden_dim in layer_dims:
            layers.append(nn.Linear(in_dim, layer_hidden_dim))
            layers.append(nn.LayerNorm(layer_hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = layer_hidden_dim

        self.hidden_layers = nn.Sequential(*layers) if layers else nn.Identity()
        output_dim = num_classes if task_type == "classification" else num_parameters
        self.output_layer = nn.Linear(in_dim, output_dim)
        self.decoder = nn.Identity()
        self.hyperparameters = {
            "embedding_dim": embedding_dim,
            "num_parameters": num_parameters,
            "dropout_rate": dropout_rate,
            "positive_only": positive_only,
            "task_type": task_type,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "hidden_sizes": layer_dims,
            "num_classes": num_classes,
            "prediction_head_type": "mlp",
        }

    def forward(self, embeddings):
        """Return ``(outputs, outputs)``: the MLP head has no separate latent vector."""
        hidden = self.hidden_layers(embeddings)
        logits = self.output_layer(hidden)
        if self.task_type == "classification":
            return logits, logits

        predictions = logits
        if self.positive_only:
            predictions = F.softplus(predictions)
        return predictions, predictions


def _stack_session_batch(batch_sessions):
    """
    Convert a list of sessions (each containing per-task tensors) into batched
    tensors per task for efficient embedding computation.
    """
    if not batch_sessions:
        return []

    num_tasks = len(batch_sessions[0])
    task_batches = [[] for _ in range(num_tasks)]

    for session in batch_sessions:
        # Preserve per-task ordering so downstream stacking matches the task-specific models.
        for task_idx in range(num_tasks):
            task_tensor = session[task_idx]
            # Some cached task tensors carry leading singleton dims (e.g. 1x1xTxd); squeeze
            # them so stacked tensors become (batch, seq_len, feature_dim).
            while task_tensor.dim() > 2 and task_tensor.shape[0] == 1:
                task_tensor = task_tensor.squeeze(0)
            task_batches[task_idx].append(task_tensor)

    return [torch.stack(task_tensors, dim=0) for task_tensors in task_batches]


def precompute_embeddings(
    task_specific_models,
    session_task_data_x,
    max_sequence_len=1500,
    task_specific_hidden_size=128,
    fusion_type="concat",
    pool_kernel_size=64,
    pool_stride=32,
    batch_size=32,
    embeddings_to_use="last_only",
):
    """Return the frozen-encoder embeddings of all sessions, shape (num_sessions, embedding_dim).

    ``task_specific_models`` maps task name to encoder in task order, and each entry of
    ``session_task_data_x`` is a list of per-task tensors.  ``embeddings_to_use`` sets how a
    task sequence is summarised (see :class:`MultiTaskEmbedModel`).  ``fusion_type``,
    ``pool_kernel_size`` and ``pool_stride`` are only stored: the task embeddings are always
    concatenated and no pooling is applied.
    """
    embedding_model = MultiTaskEmbedModel(
        models=list(task_specific_models.values()),
        max_len=max_sequence_len,
        fusion_type=fusion_type,
        task_specific_hidden_size=task_specific_hidden_size,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        embeddings_to_use=embeddings_to_use,
    ).to(COMPUTE_DEVICE)

    # Freeze the encoders so the precomputation tracks no gradients.
    for param in embedding_model.parameters():
        param.requires_grad = False

    embedding_model.eval()

    all_embeddings = []

    # Process in batches to manage memory
    num_batches = (len(session_task_data_x) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(session_task_data_x))
            batch_sessions = session_task_data_x[start_idx:end_idx]

            batched_task_inputs = _stack_session_batch(batch_sessions)
            if not batched_task_inputs:
                continue

            batch_embeddings = embedding_model(batched_task_inputs)
            # Move to CPU before concatenation so GPU memory stays bounded across many chunks.
            all_embeddings.append(batch_embeddings.detach().cpu())

    if not all_embeddings:
        raise ValueError("No embeddings were computed; check the provided session data.")

    return torch.cat(all_embeddings, dim=0)


class EmbeddingDataset(Dataset):
    """
    A PyTorch Dataset for embedding-based hierarchical model training.
    This provides efficient batch loading and shuffling capabilities.
    """

    def __init__(self, embeddings, y_values, item_level_data=None, ids=None):
        """Wrap row-aligned embeddings and targets, plus optional item-level data and sample IDs."""
        self.embeddings = embeddings
        self.y_values = y_values
        self.item_level_data = item_level_data
        self.ids = ids

        # Validate shapes
        assert len(embeddings) == len(y_values), "Number of embeddings and y_values must match"
        if item_level_data is not None:
            assert len(embeddings) == len(item_level_data), (
                "Number of embeddings and item_level_data must match"
            )
        if ids is not None:
            assert len(embeddings) == len(ids), "Number of embeddings and ids must match"

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        """Return one sample as a dict of ``"embeddings"``, ``"y_values"`` and optional fields."""
        sample = {"embeddings": self.embeddings[idx], "y_values": self.y_values[idx]}

        if self.item_level_data is not None:
            sample["item_level_data"] = self.item_level_data[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def collate_fn(batch):
    """Batch the per-sample dicts of :class:`EmbeddingDataset`."""
    # Extract all keys from the first sample
    keys = batch[0].keys()

    # Initialize batched data
    batched_data = {}

    for key in keys:
        if key == "id":
            # Handle IDs as a list
            batched_data[key] = [sample[key] for sample in batch]
        else:
            # Handle tensors by stacking
            batched_data[key] = torch.stack([sample[key] for sample in batch])

    return batched_data


class GroupedBatchSampler(Sampler):
    """Yield precomputed batches of row indices."""

    def __init__(self, batches):
        self.batches = [list(batch) for batch in batches if batch]

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def _build_grouped_row_batches(
    sample_ids,
    batch_size,
    shuffle=False,
    seed=0,
    allowed_indices=None,
):
    """Group full IDs into batches so each ID stays within a single batch."""
    if sample_ids is None:
        raise ValueError("sample_ids are required for grouped row batching.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0 for grouped row batching.")

    base_indices = (
        list(range(len(sample_ids)))
        if allowed_indices is None
        else [int(idx) for idx in allowed_indices]
    )
    grouped_indices = {}
    ordered_ids = []
    for row_idx in base_indices:
        sample_id = sample_ids[row_idx]
        if sample_id not in grouped_indices:
            grouped_indices[sample_id] = []
            ordered_ids.append(sample_id)
        grouped_indices[sample_id].append(row_idx)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered_ids)

    batches = []
    current_batch = []
    current_batch_rows = 0
    for sample_id in ordered_ids:
        row_indices = grouped_indices[sample_id]
        if current_batch and current_batch_rows + len(row_indices) > batch_size:
            batches.append(current_batch)
            current_batch = []
            current_batch_rows = 0

        current_batch.extend(row_indices)
        current_batch_rows += len(row_indices)

        if current_batch_rows >= batch_size:
            batches.append(current_batch)
            current_batch = []
            current_batch_rows = 0

    if current_batch:
        batches.append(current_batch)

    return batches


def _group_row_indices_by_id(sample_ids):
    """Return insertion-ordered row indices grouped by sample ID."""
    grouped_indices = {}
    ordered_ids = []
    for row_idx, sample_id in enumerate(sample_ids):
        if sample_id not in grouped_indices:
            grouped_indices[sample_id] = []
            ordered_ids.append(sample_id)
        grouped_indices[sample_id].append(row_idx)
    return ordered_ids, grouped_indices


def aggregate_classification_targets_by_id(y_true, sample_ids):
    """Collapse row-level classification targets to one target per sample ID."""
    if sample_ids is None:
        raise ValueError("sample_ids are required for player-level aggregation.")
    target_tensor = _ensure_tensor_labels(y_true).view(-1).long()
    if target_tensor.shape[0] != len(sample_ids):
        raise ValueError("sample_ids must be row-aligned with y_true.")

    ordered_ids, grouped_indices = _group_row_indices_by_id(sample_ids)
    aggregated_targets = []
    for sample_id in ordered_ids:
        row_indices = grouped_indices[sample_id]
        sample_targets = target_tensor[row_indices]
        if sample_targets.numel() == 0:
            continue
        first_target = sample_targets[0]
        if not torch.equal(sample_targets, first_target.expand_as(sample_targets)):
            raise ValueError(
                "Conflicting classification targets found for sample_id='{}'.".format(sample_id)
            )
        aggregated_targets.append(first_target)

    if not aggregated_targets:
        return torch.empty(0, dtype=torch.long), []

    return torch.stack(aggregated_targets, dim=0), ordered_ids


def aggregate_classification_logits_by_id(logits, y_true, sample_ids):
    """Average classification logits within each sample ID group."""
    if sample_ids is None:
        raise ValueError("sample_ids are required for player-level aggregation.")
    if logits.shape[0] != len(sample_ids):
        raise ValueError("sample_ids must be row-aligned with logits.")

    target_tensor = _ensure_tensor_labels(y_true).view(-1).long().to(logits.device)
    if target_tensor.shape[0] != logits.shape[0]:
        raise ValueError("y_true must be row-aligned with logits.")

    ordered_ids, grouped_indices = _group_row_indices_by_id(sample_ids)
    aggregated_logits = []
    aggregated_targets = []
    for sample_id in ordered_ids:
        row_indices = grouped_indices[sample_id]
        idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=logits.device)
        sample_logits = logits.index_select(0, idx_tensor)
        sample_targets = target_tensor.index_select(0, idx_tensor)
        if sample_targets.numel() == 0:
            continue
        first_target = sample_targets[0]
        if not torch.equal(sample_targets, first_target.expand_as(sample_targets)):
            raise ValueError(
                "Conflicting classification targets found for sample_id='{}'.".format(sample_id)
            )
        aggregated_logits.append(sample_logits.mean(dim=0))
        aggregated_targets.append(first_target)

    if not aggregated_logits:
        return (
            torch.empty((0, logits.shape[1]), dtype=logits.dtype, device=logits.device),
            torch.empty(0, dtype=torch.long, device=logits.device),
            [],
        )

    return (
        torch.stack(aggregated_logits, dim=0),
        torch.stack(aggregated_targets, dim=0),
        ordered_ids,
    )


def _sample_train_indices_by_id(
    sample_ids,
    samples_per_id,
    sampling_seed,
):
    """Return a reproducible subset of row indices grouped by sample ID."""
    if samples_per_id <= 0:
        return list(range(len(sample_ids)))

    grouped_indices = {}
    for row_idx, sample_id in enumerate(sample_ids):
        grouped_indices.setdefault(sample_id, []).append(row_idx)

    rng = random.Random(sampling_seed)
    sampled_indices = []
    for row_indices in grouped_indices.values():
        if len(row_indices) <= samples_per_id:
            sampled_indices.extend(row_indices)
            continue
        sampled_indices.extend(sorted(rng.sample(row_indices, samples_per_id)))
    return sampled_indices


def train_embedding_hierarchical_model(
    model,
    train_embeddings,
    train_y,
    train_item_level_data,
    val_embeddings,
    val_y,
    val_item_level_data,
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    args=None,
    patience=10,
    eval_interval=2,
    train_sample_ids=None,
    val_sample_ids=None,
    train_samples_per_id_per_epoch=0,
    sampling_seed=42,
):
    """Train a PLVM head on precomputed frozen-encoder embeddings.

    Returns ``(model, training_losses, validation_losses)``.  Validation runs every
    ``eval_interval`` epochs; ``patience`` counts evaluations, and early stopping also needs
    ``args.use_early_stopping``.  ``args`` carries the loss type, L2 weight and the other
    run settings.

    ``train_samples_per_id_per_epoch > 0`` resamples that many rows per training sample ID
    each epoch (seeded by ``sampling_seed``).  With ``args.aggregate_classification_by_player``
    in classification runs, each sample ID stays within one batch and its logits are averaged
    before the cross-entropy.
    """

    aggregate_classification_by_player = bool(
        getattr(args, "aggregate_classification_by_player", False)
        and getattr(args, "loss_type", "") == "classification"
    )
    use_grouped_resampling = train_samples_per_id_per_epoch > 0
    if use_grouped_resampling:
        if train_sample_ids is None:
            raise ValueError(
                "train_sample_ids must be provided when train_samples_per_id_per_epoch > 0."
            )
        if len(train_sample_ids) != len(train_embeddings):
            raise ValueError("train_sample_ids must be row-aligned with train_embeddings.")
        grouped_train_indices = {}
        for row_idx, sample_id in enumerate(train_sample_ids):
            grouped_train_indices.setdefault(sample_id, []).append(row_idx)
        expected_train_rows_per_epoch = sum(
            min(train_samples_per_id_per_epoch, len(row_indices))
            for row_indices in grouped_train_indices.values()
        )
    else:
        grouped_train_indices = None
        expected_train_rows_per_epoch = len(train_embeddings)

    if aggregate_classification_by_player:
        if train_sample_ids is None or len(train_sample_ids) != len(train_embeddings):
            raise ValueError(
                "train_sample_ids must be row-aligned with train_embeddings when "
                "aggregate_classification_by_player is enabled."
            )
        if val_sample_ids is None or len(val_sample_ids) != len(val_embeddings):
            raise ValueError(
                "val_sample_ids must be row-aligned with val_embeddings when "
                "aggregate_classification_by_player is enabled."
            )

    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(
        embeddings=train_embeddings,
        y_values=train_y,
        item_level_data=train_item_level_data,
        ids=train_sample_ids if aggregate_classification_by_player else None,
    )

    val_dataset = EmbeddingDataset(
        embeddings=val_embeddings,
        y_values=val_y,
        item_level_data=val_item_level_data,
        ids=val_sample_ids if aggregate_classification_by_player else None,
    )

    # Create DataLoaders with shuffling for training
    train_loader = None
    if not use_grouped_resampling and not aggregate_classification_by_player:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # tensors are already in memory; worker processes add no benefit
            pin_memory=False,  # callers pass tensors already on the training device
        )
    elif aggregate_classification_by_player:
        initial_train_batches = _build_grouped_row_batches(
            train_sample_ids,
            batch_size=batch_size,
            shuffle=True,
            seed=sampling_seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=GroupedBatchSampler(initial_train_batches),
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )

    if aggregate_classification_by_player:
        val_batches = _build_grouped_row_batches(
            val_sample_ids,
            batch_size=batch_size,
            shuffle=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=GroupedBatchSampler(val_batches),
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,  # callers pass tensors already on the training device
        )

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.25), int(num_epochs * 0.5), int(num_epochs * 0.75)],
        gamma=0.5,
    )

    # Setting the model to training mode
    model.train()

    # Lists to store training and validation losses
    training_losses = []
    training_l2_loss = []
    validation_losses = []

    # Training loop
    best_val_loss = float("inf")
    best_validation_epoch = 0
    epochs_without_improvement = 0
    last_epoch_train_loss = float("nan")
    last_epoch_val_loss = float("nan")

    # Log data statistics
    logging.info(
        f"Train Embeddings Mean: {train_embeddings.mean()}, Val Embeddings Mean: {val_embeddings.mean()}"
    )
    logging.info(
        f"Train Embeddings Std: {train_embeddings.std()}, Val Embeddings Std: {val_embeddings.std()}"
    )
    logging.info(
        f"Train Embeddings Shape: {train_embeddings.shape}, Val Embeddings Shape: {val_embeddings.shape}"
    )
    logging.info(f"Number of validation batches: {len(val_loader)}")
    if aggregate_classification_by_player:
        logging.info(
            "Player-aggregated classification enabled for loss/validation: "
            "rows are batched by sample ID and logits are averaged per ID."
        )
    if use_grouped_resampling:
        logging.info(
            "Grouped train resampling enabled: %d rows per ID per epoch across %d IDs "
            "(%d full rows -> %d sampled rows/epoch). Validation remains exhaustive.",
            train_samples_per_id_per_epoch,
            len(grouped_train_indices),
            len(train_embeddings),
            expected_train_rows_per_epoch,
        )
    else:
        logging.info(f"Number of training batches: {len(train_loader)}")

    for epoch in range(num_epochs):
        model.train()
        epoch_base_loss = 0.0
        epoch_l2_loss = 0.0
        num_batches = 0

        if use_grouped_resampling:
            sampled_train_indices = _sample_train_indices_by_id(
                train_sample_ids,
                train_samples_per_id_per_epoch,
                sampling_seed + epoch,
            )
            if aggregate_classification_by_player:
                train_batches = _build_grouped_row_batches(
                    train_sample_ids,
                    batch_size=batch_size,
                    shuffle=True,
                    seed=sampling_seed + epoch,
                    allowed_indices=sampled_train_indices,
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=GroupedBatchSampler(train_batches),
                    collate_fn=collate_fn,
                    num_workers=0,
                    pin_memory=False,
                )
            else:
                train_loader = DataLoader(
                    Subset(train_dataset, sampled_train_indices),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=0,
                    pin_memory=False,
                )
            if epoch == 0:
                logging.info(
                    "Epoch-wise train resampling created %d sampled rows (%d batches).",
                    len(sampled_train_indices),
                    len(train_loader),
                )
        elif aggregate_classification_by_player:
            train_batches = _build_grouped_row_batches(
                train_sample_ids,
                batch_size=batch_size,
                shuffle=True,
                seed=sampling_seed + epoch,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=GroupedBatchSampler(train_batches),
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
            )

        # Training loop using DataLoader
        for batch_data in train_loader:
            # Per-batch update: clear gradients before this batch's backward pass
            optimizer.zero_grad()

            # Extract batch data
            # The loader does not move tensors; they stay on the device the caller used.
            batch_embeddings = batch_data["embeddings"]
            batch_y = batch_data["y_values"]
            batch_ids = batch_data.get("id")

            # Forward pass on the precomputed embeddings
            latent_vector, predictions = model(batch_embeddings)
            produces_latent = getattr(model, "produces_latent", True)

            if produces_latent:
                latent_for_loss = latent_vector
            else:
                latent_for_loss = predictions

            # Compute the main loss
            if aggregate_classification_by_player:
                loss = compute_grouped_classification_loss(
                    model,
                    predictions,
                    batch_y,
                    batch_ids,
                    lambda_l2=args.lambda_l2,
                    num_classes=getattr(model, "num_classes", 2),
                )
            else:
                loss = compute_loss(
                    args.loss_type,
                    model,
                    latent_for_loss,
                    batch_y,
                    latent_dim=args.latent_dim,
                    max_kld_loss=args.max_kld_loss,
                    loss_weights=getattr(args, "loss_weight_vector", None),
                )

            # Apply a single shared L2 penalty across model types.
            l2_penalty = torch.tensor(0.0, device=batch_embeddings.device)
            if args.lambda_l2 > 0:
                l2_penalty = compute_parameter_l2_penalty(model)
                loss += args.lambda_l2 * l2_penalty

            loss.backward()

            # Apply gradient clipping
            clip_value = args.grad_clip_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            if torch.isnan(loss).any():
                print("NaN loss detected, stopping training.")
                break

            # Step the optimizer once per batch
            optimizer.step()

            # Accumulate the batch loss
            epoch_base_loss += loss.item()
            epoch_l2_loss += args.lambda_l2 * l2_penalty.item()
            num_batches += 1

        # Calculate average training loss for the epoch
        avg_epoch_l2_loss = epoch_l2_loss / num_batches
        avg_epoch_loss = epoch_base_loss / num_batches - avg_epoch_l2_loss

        training_losses.append(avg_epoch_loss)
        training_l2_loss.append(avg_epoch_l2_loss)
        # Step the scheduler
        scheduler.step()

        # Perform validation and logging every eval_interval epochs
        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            avg_val_loss = validate_embedding_model_with_dataloader(model, val_loader, args)
            validation_losses.append(avg_val_loss)
            last_epoch_train_loss = float(avg_epoch_loss)
            last_epoch_val_loss = float(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_validation_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience and args.use_early_stopping:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Log the losses
            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, "
                f"L2 Loss: {avg_epoch_l2_loss:.4f}"
            )

    # Keep the final model state; track best-validation metadata separately.
    model.best_validation_loss = float(best_val_loss)
    model.best_validation_epoch = int(best_validation_epoch)
    model.last_epoch_train_loss = float(last_epoch_train_loss)
    model.last_epoch_val_loss = float(last_epoch_val_loss)

    # Plot the training and validation loss curves
    plot_loss_curves(
        training_losses,
        validation_losses,
        [],
        num_epochs,
        args,
        eval_interval=eval_interval,
    )

    return model, training_losses, validation_losses


def validate_embedding_model_with_dataloader(model, val_loader, args):
    """Return the mean validation loss of ``model`` over the batches of ``val_loader``."""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    aggregate_classification_by_player = bool(
        getattr(args, "aggregate_classification_by_player", False)
        and getattr(args, "loss_type", "") == "classification"
    )

    with torch.no_grad():
        for batch_data in val_loader:
            # Extract batch data
            # The loader does not move tensors; they stay on the device the caller used.
            val_batch_embeddings = batch_data["embeddings"]
            val_batch_y = batch_data["y_values"]
            val_batch_ids = batch_data.get("id")

            # Forward pass with embeddings
            latent_vector, predictions = model(val_batch_embeddings)
            produces_latent = getattr(model, "produces_latent", True)
            latent_for_loss = latent_vector if produces_latent else predictions

            # Compute the loss
            if aggregate_classification_by_player:
                batch_loss = compute_grouped_classification_loss(
                    model,
                    predictions,
                    val_batch_y,
                    val_batch_ids,
                    lambda_l2=args.lambda_l2,
                    num_classes=getattr(model, "num_classes", 2),
                )
            else:
                batch_loss = compute_loss(
                    args.loss_type,
                    model,
                    latent_for_loss,
                    val_batch_y,
                    latent_dim=args.latent_dim,
                    max_kld_loss=args.max_kld_loss,
                    loss_weights=getattr(args, "loss_weight_vector", None),
                )

            val_loss += batch_loss.item()
            num_batches += 1

    avg_val_loss = val_loss / num_batches
    return avg_val_loss


def plot_parameter_densities(
    ground_truth,
    predictions,
    save_path,
    title="Parameter Densities",
    parameter_labels=None,
    unit_labels=["Value", "Density"],
):
    """Save ground-truth and predicted density plots, one panel per output, to ``save_path``.

    ``parameter_labels`` gives one label per output; ``unit_labels`` is ``[x_label, y_label]``.
    """
    # Create figure for parameter densities
    n_params = ground_truth.shape[1]
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    # Generate parameter labels if not provided
    if parameter_labels is None:
        parameter_labels = [f"Parameter {i + 1}" for i in range(n_params)]

    # Plot density for each parameter
    for i in range(n_params):
        # Get non-NaN values for ground truth
        gt_data = ground_truth[:, i].cpu().numpy()
        gt_data = gt_data[~np.isnan(gt_data)]

        # Get predicted values
        pred_data = predictions[:, i].detach().cpu().numpy()

        # Plot densities
        sns.kdeplot(data=gt_data, ax=axes[i], label="Ground Truth", color="blue")
        sns.kdeplot(data=pred_data, ax=axes[i], label="Predictions", color="red")

        axes[i].set_title(parameter_labels[i])
        axes[i].set_xlabel(unit_labels[0])
        axes[i].set_ylabel(unit_labels[1])
        axes[i].legend()

    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_curves(
    training_losses,
    validation_losses,
    training_contrastive_loss,
    num_epochs,
    args,
    eval_interval=10,
    y_lim=[0, 50],
):
    """
    Plot the training and validation loss curves and save them to
    ``<args.log_dir>/training_validation_loss_curve.pdf``.

    ``validation_losses`` holds one value per evaluation, made after every
    ``eval_interval``-th epoch (0-based ``epoch % eval_interval == 0``) and after the last
    epoch, which is how :func:`train_embedding_hierarchical_model` validates.
    ``training_contrastive_loss`` is an optional extra per-epoch curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(training_losses) + 1),
        training_losses,
        label=f"Training Loss (Final: {training_losses[-1]:.4f})",
    )
    # 1-based epoch numbers of the validation passes.
    validation_epochs = [
        epoch + 1
        for epoch in range(len(training_losses))
        if epoch % eval_interval == 0 or epoch == num_epochs - 1
    ][: len(validation_losses)]
    plt.plot(
        validation_epochs,
        validation_losses,
        label=f"Validation Loss (Final: {validation_losses[-1]:.4f})",
    )
    if training_contrastive_loss:
        plt.plot(
            range(1, len(training_contrastive_loss) + 1),
            training_contrastive_loss,
            label=f"Structure Loss (Final: {training_contrastive_loss[-1]:.4f})",
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Hierarchical Loss Curves - Model ID: {args.model_name}")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_lim)

    # Annotate the key parameters
    textstr = (
        f"Num Epochs: {num_epochs}\n"
        f"Batch Size: {args.batch_size}\n"
        f"Learning Rate: {args.learning_rate:.1e}\n"
        f"Seq Length: {args.seq_length}\n"
        f"Encoder Hidden Size: {args.encoder_hidden_size}\n"
        f"Latent Dim: {args.latent_dim}\n"
        f"Decoder Hidden Dim: {args.decoder_hidden_dim}\n"
        f"Lambda L2: {args.lambda_l2}\n"
        f"Grad Clip Norm: {args.grad_clip_norm}\n"
        f"Loss Type: {args.loss_type}"
    )
    # Figure coordinates: 20% from the left, 60% from the bottom.
    plt.gcf().text(0.2, 0.6, textstr, fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

    plt.savefig(f"{args.log_dir}/training_validation_loss_curve.pdf")
    plt.close()
