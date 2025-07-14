 # SBIN: Self‑Learned Brachistochrone Informed Network
 Abstract
 We introduce a hybrid deep model that combines temporal convolutional networks (TCNs) and Transformer
 encoders for modeling multi-feature time series subject to physical laws. Our architecture processes
 arbitrary CSV inputs (with a 
2
 1
 time column) through parallel TCN and self-attention branches, then fuses
 their features. We embed physics-informed losses by modeling both Lagrangian and Hamiltonian
 formulations: a learnable mass matrix $M(x)$ and potential $U(x)$ define a Lagrangian $L=T-U$, whose
 Euler–Lagrange equation is enforced; similarly, the Hamiltonian $H=T+U$ imposes $\dot{q}=\partial H/
 \partial p,\;\dot{p}=-\partial H/\partial q$ constraints . In addition, we introduce a novel Brachistochrone
inspired path-integral loss that penalizes non-time-optimal trajectories (analogous to the classical fastest
descent problem ). To respect geometric consistency, we derive a relation tensor $R$ from unsupervised
 KMeans clusters and add a regularizer encouraging nearby (same-cluster) points to have similar latent
 dynamics. The total loss combines data fidelity with these physics priors. In experiments on synthetic
 physics-governed datasets (e.g. multi-body oscillators), our model achieves lower trajectory error and better
 energy conservation than baselines. We visualize 3D trajectories of the predicted states and highlight the
 energy-minimizing paths. All algorithms are described in detail with pseudocode. 
Introduction
 1
 3
 4
 Accurate modeling of complex time series often requires capturing long-range dependencies and
 respecting underlying physics. Traditional recurrent models (LSTM/GRU) can struggle with vanishing
 gradients, whereas Temporal Convolutional Networks (TCNs) using dilated causal convolutions have shown
 superior long-range performance . Meanwhile, Transformer encoders with multi-head self-attention can
 learn both short- and long-term temporal correlations . To leverage physical laws, Physics-Informed
 Neural Networks (PINNs) embed differential equations into the training loss . Prior work has successfully
 imposed Lagrangian or Hamiltonian structure: Lagrangian Neural Networks (LNNs) parameterize arbitrary
 Lagrangians via neural nets , and Hamiltonian Neural Networks (HNNs) enforce energy conservation by
 learning a Hamiltonian function that satisfies $\dot{q}=\partial H/\partial p,\;\dot{p}=-\partial H/\partial
 q$ . 
6
 5
 In this work we integrate these ideas into a single model for multivariate time series. Our network ingests
 any CSV with a 
time column and $D$ features. It uses a hybrid TCN+Transformer encoder to extract
 temporal features, then applies physics-based losses. We learn a state-dependent mass matrix $M(x)$ and
 potential energy $U(x)$ so that the system’s Lagrangian $L(x,\dot x)=\frac12\dot x^\top M(x)\dot x - U(x)$
 and Hamiltonian $H(x,p)=\frac12p^\top M(x)^{-1}p + U(x)$ govern dynamics. We penalize deviation from the
 Euler–Lagrange equations and from canonical Hamilton’s equations (equation (2) below) in the loss. To
 encourage time-optimal (least-action) behavior, we add a Brachistochrone-inspired path loss that integrates
 travel-time cost along the feature trajectory . Finally, we perform KMeans clustering on the data and
 build a relation tensor $R$ (with $R_{ij}=1$ if points $i,j$ share a cluster) to softly enforce that same-cluster
 2
 1
points follow consistent geometry. The result is a hybrid physics-informed model that generalizes across
 arbitrary feature sets. 
Our contributions are: (1) A hybrid TCN-Transformer architecture for time series which easily adapts to any
 number of feature columns; (2) simultaneous Lagrangian and Hamiltonian losses with learnable mass
 and potential, grounding the model in classical mechanics; (3) a novel Brachistochrone path-integral loss
 for time-optimal trajectories; (4) a cluster-based relation regularizer imposing geometric consistency; (5)
 end-to-end training pseudocode. In experiments, we demonstrate that our approach reduces prediction
 error and conserves physical invariants better than non-physics baselines. 
Related Work
 Deep time-series models range from RNNs to convolutional and attention-based networks. Bai et al.
 introduced the TCN, a purely convolutional sequence model using dilated causal convolutions with residual
 connections, which outperforms recurrent models on various tasks . Transformer-style self-attention has
 also been adapted to time series, leveraging multi-head attention to capture long-range dependencies .
 However, these models typically ignore domain physics. 
3
 5
 1
 4
 Physics-informed networks embed known laws into learning. Raissi et al. proposed PINNs, which train
 neural networks to satisfy given PDEs as soft constraints . In mechanical systems, Lagrangian and
 Hamiltonian formulations have been used: Lagrangian Neural Networks learn a network $L_\theta(q,\dot q)$
 so that the resulting Euler–Lagrange ODE matches data, even in arbitrary coordinates . Deep Lagrangian
 Networks (DeLaN) impose the Euler–Lagrange equation explicitly to ensure energy-conserving dynamics
 . Hamiltonian Neural Networks (HNNs) learn a Hamiltonian $H_\theta(q,p)$ so that $\dot q=\partial H/
 \partial p$, $\dot p=-\partial H/\partial q$ hold; this guarantees conservation of the learned “energy”
 quantity
 6
 1
 . Our method blends both approaches by using both a Lagrangian and a Hamiltonian loss. 
6
 Lastly, unsupervised clustering has been used to extract relational structure in data. While not common in
 PINNs, ideas like graph Laplacian regularization suggest enforcing similarity of known-related points. We
 adopt a simple relation tensor from KMeans: points in the same cluster receive a quadratic penalty on their
 prediction differences. This encourages geometric consistency among locally similar data points during
 training. 
Methodology
 Data Input and Preprocessing
 Our model accepts an arbitrary CSV file with a time column $t$ and $D$ feature columns ${x_i}$. We first
 normalize time and each feature to zero mean and unit variance. We then form training sequences $X_{0:T}
 \in \mathbb{R}^{T\times D}$ of length $T$ (by sliding windows or full-trajectory sampling) and
 corresponding derivatives $\dot X_{0:T}$ (computed by finite differences or provided). These sequences
 feed into the network detailed below. 
2
Hybrid TCN-Transformer Architecture
 The network processes the input sequence $X = [x_0, x_1, \dots, x_{T-1}]$ in parallel through two branches:
 • 
• 
3
 TCN branch: A stack of 1D convolutional layers with exponentially increasing dilation factors and
 causal padding, as in Bai et al. . Specifically, each TCN block has a dilated causal Conv1D (kernel
 size $k$, dilation $d$), followed by ReLU and dropout. Successive layers double $d$, covering longer
 history. The TCN outputs $F_{\text{TCN}}\in\mathbb{R}^{T\times K}$, capturing local temporal
 patterns.
 4
 Transformer encoder branch: A Transformer encoder (as in “Attention Is All You Need”) with $H$
 multi-head self-attention layers and positional encoding . This branch outputs $F_{\text{Tr}}
 \in\mathbb{R}^{T\times K}$ of the same shape. The multi-head self-attention allows each time step
 to attend to all others, capturing global dependencies.
 We then fuse these features, for example by concatenation along the feature axis or by learned linear
 projection. Let $Z = \mathrm{Concat}(F_{\text{TCN}}, F_{\text{Tr}})\in \mathbb{R}^{T\times (2K)}$. A final
 feedforward head (e.g. MLP or linear layer) maps $Z$ to predicted state derivatives $\hat{\dot X}
 \in\mathbb{R}^{T\times D}$ and predicted canonical momenta $\hat{P}\in\mathbb{R}^{T\times D}$ (see
 physics losses below). 
