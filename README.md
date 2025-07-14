# SBIN
Self-learned Brachistochrone Informed Network (SBIN) by Ashifur Rahman (BSC in physics at Khulna university)
# SBIN: Sequential Brachistochrone Informed Network

SBIN (Sequential Brachistochrone Informed Network) is a physics-informed neural network (PINN) designed to learn optimal trajectories, particularly the brachistochrone curve (fastest descent under gravity):contentReference[oaicite:15]{index=15}. By embedding the governing physics (e.g. energy conservation) into the training loss, SBIN can efficiently find minimum-time paths without extensive random exploration:contentReference[oaicite:16]{index=16}. The network combines convolutional and Transformer layers to process sequential inputs, and uses automatic differentiation to enforce physics constraints.

## Key Features

- **Physics-Informed Loss:** Incorporates physics laws and goals into the loss function (e.g. conserving energy and minimizing time):contentReference[oaicite:17]{index=17}.
- **Hybrid CNN-Transformer:** A 1D CNN extracts local features from the input sequence, and a Transformer encoder models long-range dependencies:contentReference[oaicite:18]{index=18}.
- **Brachistochrone Minimization:** Trains on the classic brachistochrone problem. The analytic solution is a cycloid curve:contentReference[oaicite:19]{index=19}, and SBIN learns this path by minimizing the physics-based loss.
- **Relation Tensors:** Computes pairwise relations (a tensor of interactions) between trajectory points to capture structural dependencies.
- **End-to-End Training:** Simple setup; run `sbin_model.py` in Colab or locally after installing dependencies.

:contentReference[oaicite:20]{index=20} *Figure. SBIN network architecture overview. The input domain variable (time *t*) is passed through CNN + Transformer layers (target function \u03b8) to produce output trajectory points *u*. Physics-informed losses (right) combine governing laws, boundary constraints, and the goal (shortest time):contentReference[oaicite:21]{index=21}.* The model maps input domain variables (e.g. normalized time) to output design variables (trajectory coordinates):contentReference[oaicite:22]{index=22}. CNN layers capture local structure, the Transformer encodes global sequence relations, and learned relation tensors encode interactions between points. Automatic differentiation computes gradients of the physics-informed loss (governing equation, constraints, goal) during training.

