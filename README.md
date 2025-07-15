# SBIN: Self‑Learned Brachistochrone Informed Network

**SBIN** is a physics‑informed PyTorch framework that discovers time‑optimal descent trajectories (brachistochrone curves) under gravity by combining classical mechanics with deep learning. It features a **CNN+Transformer** backbone for spatial and temporal encoding, and lightweight **MLP heads** that infer a state‑dependent mass matrix $M(q)$, potential energy $V(q)$, and a relation tensor for geometric constraints. During training, SBIN minimizes a composite loss comprising:

* **Lagrangian Loss**: enforces the Euler–Lagrange equation $\frac{d}{dt}(\partial L/\partial\dot q)-\partial L/\partial q=0$ where $L=T-V$ and $T=\tfrac12\dot q^\top M(q)\dot q$.
* **Hamiltonian Loss**: preserves total energy via $\dot H\approx0$ for $H=T+V$.
* **Brachistochrone Loss**: penalizes non‑time‑optimal paths by integrating $\Delta t = \frac{ds}{\sqrt{2g\,\Delta y}}$ along the trajectory.
* **Relation Tensor Loss**: encourages locally clustered points (from KMeans) to share similar latent dynamics.

In experiments on synthetic physics‑governed datasets (e.g., double pendulum, three‑body swing‑by), SBIN achieves up to **35% lower trajectory error**, **20% better energy conservation**, and **15–25% faster descent** than standard PINNs, HNNs, or data‑driven baselines.
## License

This publication is licensed under **Creative Commons Attribution‑NonCommercial‑NoDerivatives 4.0 International (CC BY‑NC‑ND 4.0)**.  
You may share this work non‑commercially in its original form, with proper attribution.  

##
