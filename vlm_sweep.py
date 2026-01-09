"""
W&B Sweep training script for VLM with Mixture of Modality Heads.
Run with: wandb agent <sweep_id>
"""
import os
import wandb

# Import everything from vlm_playground
from vlm_playground import (
    # Config and model
    VLMConfig, VisionLanguageModel,
    # Training utilities
    train_vlm, evaluate_vlm, get_vlm_wandb_config, get_batch,
    maybe_compile, autocast_context,
    # Tokenizer
    vocab_size, stoi, itos, pad_idx, bos_idx, eos_idx,
    # Device and settings
    device, use_fused_optimizer, use_wandb,
    # torch
    torch,
)


def train_sweep():
    """Single sweep trial - reads hyperparams from wandb.config"""
    # Initialize wandb run (sweep agent handles this)
    run = wandb.init()

    # Get hyperparameters from sweep
    sweep_config = wandb.config
    learning_rate = sweep_config.get('learning_rate', 2e-3)
    batch_size = sweep_config.get('batch_size', 64)

    # Model capacity params (with defaults for backward compatibility)
    embed_size = sweep_config.get('embed_size', 640)
    head_num = sweep_config.get('head_num', 10)
    layer_num = sweep_config.get('layer_num', 6)
    vit_num_layers = sweep_config.get('vit_num_layers', 4)
    vit_num_heads = sweep_config.get('vit_num_heads', 8)
    image_embed_dim = sweep_config.get('image_embed_dim', 512)

    # Mixture of Modality Heads params
    head_pct_vision = sweep_config.get('head_pct_vision', 0.4)
    head_pct_text = sweep_config.get('head_pct_text', 0.4)

    # Ensure embed_size is divisible by head_num
    if embed_size % head_num != 0:
        head_num = 10  # Fallback to safe value

    # Ensure image_embed_dim is divisible by vit_num_heads
    if image_embed_dim % vit_num_heads != 0:
        vit_num_heads = 4  # Fallback to safe value

    # Calculate head distribution for logging
    h_v = int(head_num * head_pct_vision)
    h_t = int(head_num * head_pct_text)
    h_vt = head_num - h_v - h_t

    print(f"Trial: lr={learning_rate:.2e}, bs={batch_size}, "
          f"embed={embed_size}, heads={head_num}, layers={layer_num}, "
          f"vit_layers={vit_num_layers}, vit_heads={vit_num_heads}, img_dim={image_embed_dim}")
    print(f"Head allocation: V={h_v}, T={h_t}, VT={h_vt} "
          f"(pct: {head_pct_vision:.0%}/{head_pct_text:.0%}/{1-head_pct_vision-head_pct_text:.0%})")

    # Create config with sweep hyperparameters
    config = VLMConfig(
        img_size=96,
        patch_size=16,
        image_embed_dim=image_embed_dim,
        vit_num_layers=vit_num_layers,
        vit_num_heads=vit_num_heads,
        vocab_size=vocab_size,
        embed_size=embed_size,
        seq_len=256,
        head_num=head_num,
        layer_num=layer_num,
        head_pct_vision=head_pct_vision,
        head_pct_text=head_pct_text,
        batch_size=batch_size,
        total_steps=500,  # Reduced for faster trials
        learning_rate=learning_rate,
        val_interval=50,
        checkpoint_interval=1000,  # Disable checkpointing during sweep
    )

    # Initialize model
    model = VisionLanguageModel(config)
    model = maybe_compile(model)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        fused=use_fused_optimizer
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.total_steps,
        eta_min=1e-6
    )

    # Log config
    wandb.config.update(config.to_dict())

    # Train
    losses, val_losses = train_vlm(
        model, optimizer, scheduler, config,
        save_dir='sweep_checkpoints'
    )

    # Log final metrics
    final_val_loss = val_losses[-1] if val_losses else float('inf')
    wandb.log({
        'final_train_loss': losses[-1],
        'final_val_loss': final_val_loss,
    })

    print(f"Final: train_loss={losses[-1]:.4f}, val_loss={final_val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    train_sweep()
