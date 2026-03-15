#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# EdgeAttentionUNet — Multi-GPU DDP training script
#
# ── EDIT THESE ───────────────────────────────────────────────────────────────
N_GPUS=4          # number of GPUs  (num= in bsub -gpu flag)
N_CPU=8           # total CPU cores (bsub -n flag)  → N_CPU / N_GPUS per process
N_WORKERS=2       # DataLoader workers per GPU process
CONFIG="configs/config_edge_model_hpc_full.yaml"
CHECKPOINT=""     # path to .pt to resume from — leave empty to train from scratch
#                   e.g. CHECKPOINT="checkpoints/epoch_010_val_dice_0.8500.pt"
# ─────────────────────────────────────────────────────────────────────────────

# ── bsub launcher — built from variables above ────────────────────────────────
multi_gpu_new_b () {
    bsub -Is \
         -q gpu \
         -gpu "num=${N_GPUS}:j_exclusive=yes:gmodel=NVIDIAA30" \
         -R "osrel>=80 && ui==aiml_batch_training" \
         -P ai_warstein \
         -n ${N_CPU} \
         "$1"
}

# ── Environment ───────────────────────────────────────────────────────────────
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF

mkdir -p logs checkpoints runs

# ── Build torchrun command ────────────────────────────────────────────────────
py_sc="torchrun \
    --nproc_per_node=${N_GPUS} \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_edge_model.py \
    --config ${CONFIG} \
    --num_workers ${N_WORKERS}"

# Append --resume only when CHECKPOINT is set
if [ -n "$CHECKPOINT" ]; then
    [ ! -f "$CHECKPOINT" ] && echo "ERROR: checkpoint not found: $CHECKPOINT" && exit 1
    py_sc="${py_sc} --resume ${CHECKPOINT}"
    echo "Resuming from : $CHECKPOINT"
else
    echo "Training from scratch"
fi

echo "GPUs: ${N_GPUS} | CPU cores: ${N_CPU} | Workers/process: ${N_WORKERS}"
echo "Config: ${CONFIG}"

# ── Submit ────────────────────────────────────────────────────────────────────
multi_gpu_new_b "$py_sc"

