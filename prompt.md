# Role and Objective
You are an expert AI Research Engineer and Machine Learning Architect specializing in Graph Neural Networks (GNNs), Neuroimaging, and Representation Learning. Your target venue is ICLR. 

I am working on a research paper targeting ICLR 2027, building upon our recent AAAI 2027 submission ("MetaSFC"). 
**The Problem:** In our AAAI paper, we used a working-memory meta-analysis prior as a soft regularizer (Pearson correlation loss penalty) in a GCN backbone (MetaSFC), and as a naive feature scaler in Ridge Regression. While this vastly improved biomarker stability and task-specificity, it **failed to improve raw cognitive prediction** over a standard FC+SC Ridge baseline. Our supervisor noted that if the prior is truly informative, it should improve prediction, not just stabilize explanations.
**The Goal:** Implement novel methodological backbones that deeply integrate the meta-analysis prior into the *predictive inductive bias* of the model, rather than treating it as an auxiliary loss or post-hoc weight scaler. We must improve prediction accuracy while retaining the biomarker significance.

## Repository Context
- **GitHub Repository:** https://github.com/sanjanbaitalik/metaSFC
- **Current Architecture:** `src/metascfc/models/ms_inter_gcn.py` (GCN encoders for FC/SC, interaction bottleneck, auxiliary heads).
- **Data Pipeline:** Handled via `scripts/` and `src/metascfc/data/`. The input directories and nested cross-validation (10 seeds, 5 outer folds) must remain **strictly identical** to ensure fair comparison.
- **Prior Data:** Voxel-to-ROI mapped meta-analysis priors are available in `outputs/priors/working_memory/aal116/`.

## Your Task: Implement 3 Novel Methodological Backbones
Do not change the dataset, the AAL116 atlas, or the repeated nested CV evaluation protocol. You are to implement the following three ICLR-caliber methodologies as new model classes or training scripts, integrating them seamlessly into the existing codebase.

### Method 1: Network-Constrained Prior-Laplacian Ridge (Linear/Hybrid)
*Concept:* Instead of scaling features by the prior (which standardizes away the prior's effect), formulate a structured regularization penalty.
*Math:* Minimize $||y - X\beta||^2 + \lambda_1 ||\beta||^2 + \lambda_2 \beta^T L_{prior} \beta$
*Implementation:* 
1. Construct a Graph Laplacian ($L_{prior}$) from the meta-analysis prior map (e.g., thresholding the top 30 ROIs to form a prior adjacency matrix, then computing the Laplacian).
2. Implement a custom solver or use `scikit-learn`'s `Ridge` with a custom penalty matrix (or use `network-constrained regression` via coordinate descent). 
3. This forces the predictive weights of functionally coupled regions (according to the prior) to be smooth, directly tying the biomarker structure to the predictive weights.

### Method 2: Prior-Gated Graph Attention Network (Meta-GAT)
*Concept:* The AAAI GCN diluted the prior through standard message passing. We need to inject the prior directly into the attention mechanism.
*Math:* Modify the attention coefficients $e_{ij}$ between region $i$ and $j$:
$e_{ij} = \text{LeakyReLU}(a^T [W h_i || W h_j]) + \gamma (p_i + p_j)$
where $p_i$ is the meta-analysis prior score for ROI $i$, and $\gamma$ is a learnable or tuned temperature parameter.
*Implementation:* 
1. Create `src/metascfc/models/meta_gat.py`.
2. Replace the standard GCN convolution layers with a Prior-Gated GAT layer.
3. The prior acts as a hard inductive bias, forcing the network to route information through task-relevant regions, naturally improving prediction if those regions hold signal.

### Method 3: Two-Stage Biomarker-Guided Kernel Ridge (Hybrid)
*Concept:* Use the highly stable MetaSFC saliency maps not as an explanation, but as an explicit feature-space projector for a secondary strong predictor.
*Implementation:*
1. **Stage 1:** Run the no-prior or prior-guided MS-Inter-GCN to extract the node-level coupling saliency vector $c^{(s)}$ for each subject $s$.
2. **Stage 2:** Element-wise multiply the subject's raw FC and SC upper-triangle vectors by the stabilized saliency vector $c^{(s)}$ (or use it to compute a subject-specific Prior-Weighted Kernel Matrix $K$).
3. Train a Kernel Ridge Regression or XGBoost model on this biomarker-constrained feature space.

## Execution Instructions for the AI Agent
1. **Analyze the Codebase:** First, read `src/metascfc/models/ms_inter_gcn.py`, `src/metascfc/experiments.py`, and `scripts/08_run_aaai_matrix.py` to understand how models are instantiated and trained.
2. **Implement the Models:** Write clean, heavily commented PyTorch/Scikit-Learn code for the three methods above. Place them in logical directories (e.g., `src/metascfc/models/iclr_backbones/`).
3. **Create Execution Scripts:** Write new Python scripts in the `scripts/` directory (e.g., `scripts/40_run_network_constrained_ridge.py`, `scripts/41_run_meta_gat.py`) that utilize the *exact same* data loaders and CV splitters as the AAAI scripts.
4. **Update the README.md:** 
   - You MUST update the repository's `README.md`.
   - Add a new section titled `## ICLR 2027 Methodological Extensions`.
   - Provide clear, step-by-step terminal commands on how to run the new backbones.
   - Explain the mathematical intuition behind each new method briefly in the README.

## Constraints & Quality Assurance
- **Do not break existing AAAI reproduction scripts.** The old code must still run.
- Ensure all new models output the exact same evaluation metrics (Pearson, RMSE, MAE, Biomarker Alignment, Rank Stability) using the existing `src/metascfc/experiments.py` evaluation loops.
- Handle edge cases (e.g., isolated nodes in the prior Laplacian, attention softmax overflow).
- Use type hinting and docstrings for all new classes.

Begin by acknowledging this prompt, outlining your step-by-step plan to modify the repository, and then proceed to write the code and update the `README.md`.