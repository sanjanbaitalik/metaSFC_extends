# Role and Objective
You are an expert Neuro-AI Research Engineer. We are pivoting our ICLR 2027 paper to incorporate **LLM-Generated Semantic Priors** and a **Dual-Task Experimental Matrix**.

## Task 1: The Zero-Shot LLM Prior Generator
Create a new script `scripts/45_generate_llm_priors.py`. 
- It should take a cognitive task name (e.g., "Working Memory", "Fluid Intelligence") as an argument.
- It must use the `openai` or `ollama` python library to prompt an LLM (default to `llama3` or `gpt-4o-mini`).
- The prompt must provide the list of 116 AAL brain regions (load from `inputs/atlases/AAL116_labels.csv`) and ask the LLM to assign a continuous relevance score [0.0, 1.0] to each region based on neuroscience literature.
- Parse the JSON response, normalize it, and save it to `outputs/priors/llm/{task_name}/roi_prior.csv` matching the exact schema of our existing Neurosynth priors.

## Task 2: Dual-Task Configuration
Create new config files in `configs/iclr/`:
1. `llm_wm_prior.yaml` (Target: HCP List Sorting WM task)
2. `llm_fluid_prior.yaml` (Target: PMAT Fluid Intelligence)
Ensure the data loaders in `src/metascfc/data/` are updated to support loading the HCP `ListSort_Unadj` or `WM n-back` targets alongside `PMAT24_A_CR`.

## Task 3: LLM-Gated Attention Upgrade
Create a new model file `src/metascfc/models/llm_gated_transformer.py`.
- Implement a Cross-Modal Graph Attention layer where the attention coefficients $e_{ij}$ are biased by the LLM prior matrix $M_{LLM}$.
- $e_{ij} = \text{LeakyReLU}(a^T [W h_i || W h_j]) + \lambda \cdot (p_i + p_j)$, where $p$ is the LLM prior score.
- This will allow the model to dynamically focus on the LLM-identified subgraph, which is our strategy to surpass the linear Ridge baseline.

Execute these tasks, update the README, and ensure the new scripts integrate seamlessly with our existing nested-CV evaluation loop (`src/metascfc/experiments.py`).