Toy transformer playground focused on Kimi Delta Attention (KDA).

Notes:
- The file inlines the Triton KDA stack from `resources/flash-linear-attention` (no external imports).
- Training uses the chunk KDA kernel; fused-recurrent is kept for inference.
