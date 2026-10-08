# E1 supplement (Claude, sandbox CPU) — whole-network batched inference

- Model: stage-2 DSCNet, DRIVE, batch-1 training, lr 2e-4, seed 0, validation-selected best.pt (clean_none_b3_lr0.0002_s0_ce).
- D_whole_network_all20.json: per-image metrics for upstream sliding-window inference with tiles one at a time (orig_b1)
  vs. tiles packed 8 per batch in the original order (orig_b8), original DSConv.
  Mean over 20 test images: Dice 81.73 -> 18.32, clDice 82.05 -> 18.80, beta0 error 88.9 -> 122.0, 10.4% pixels flipped.
  orig_b1 per-image Dice equals stage-2 metrics.json exactly (max abs diff 0).
- Batch-8 outputs were computed exactly via 2-sample batches [first tile, tile j] (memory limit);
  D_pairs_equivalence.json verifies this equals a true batch-8 forward to 1.4e-6.
- B_real_layers.json: per-layer relative error on real DRIVE features (16 patches, 128x128): mean over layers 0.95-0.96 for B=2..16.
- C_gradient.json: batch-8 train-mode parameter-gradient relative error 0.458, cosine 0.906 (ROI 96, dummy target).
- e1_bug.py: script (run: python e1_bug.py <DSCNet_2D_opensource> <best.pt> <out> [parts]; E1_START=k to resume part D).
