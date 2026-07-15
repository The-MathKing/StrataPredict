# Extended Experimental Protocol

To rigorously validate the proposed framework combining discrete Ricci flow and CW complexes, we outline a comprehensive experimental protocol. This protocol aims to systematically address structural limits, adversarial resilience, model complexity, and hyperparameter sensitivity, ensuring the results meet the highest standards of empirical validation.

## 1. Manifold Diversity: Beyond Molecular Graphs
Current benchmarks heavily emphasize molecular datasets (e.g., ZINC, NCI1), which primarily consist of planar, tree-like, or small cycle structures. To test the topological limits of the dynamic CW complex framework, we propose extending the evaluation to topologically diverse, non-molecular domains:

*   **Social Networks (e.g., Reddit-Binary, COLLAB):** These graphs possess dense hub-and-spoke topologies with highly interconnected cliques. Discrete Ricci flow on these datasets will test the model's ability to alleviate severe over-squashing in highly centralized topologies.
*   **Protein-Protein Interaction Networks (PPI):** Characterized by high edge density and complex multi-scale modularity, PPI networks will demonstrate the efficacy of cellular lifting in capturing higher-order biological interactions.
*   **Traffic and Infrastructure Networks:** These spatial graphs naturally exhibit grid-like structures with distinct boundary conditions. Evaluating on these networks will validate the framework's geometric fidelity and its capacity to propagate flow across long-range bottlenecks.

By diversifying the manifold domains, we isolate the topological inductive biases of our model from domain-specific features, proving its general-purpose expressivity.

## 2. Adversarial Robustness and Structural Noise
Real-world graphs are frequently noisy or incomplete. To demonstrate the structural resilience of our framework, we will subject the model to targeted topological stress tests:

*   **Random Edge Deletion/Addition:** We will inject structural noise by randomly flipping $p \in \{5\%, 10\%, 20\%\}$ of the edges. We hypothesize that Ricci-flow-guided rewiring will dynamically recover connectivity, maintaining performance where standard Message Passing Neural Networks (MPNNs) fail.
*   **Cycle Perturbation:** Since our model explicitly lifts graphs to CW complexes based on rings/cycles, we will introduce targeted adversarial attacks that break specific $k$-cycles (e.g., severing 5-cycles or 6-cycles in molecular graphs). We will evaluate the model's robustness in predicting properties when defining topological features are corrupted.

These stress tests will formally quantify the regularization effect of incorporating higher-dimensional cells into the learning process.

## 3. Advanced Ablation Study
To isolate the contribution of specific architectural mechanisms, we will conduct an advanced ablation study expanding upon the preliminary static vs. dynamic configurations:

*   **Gating Mechanisms for Cellular Messages:** We will ablate the attention-based gating functions regulating message flow between 0-cells (nodes), 1-cells (edges), and 2-cells (faces). Configurations will include: No Gating, Scalar Gating, and Vector Gating.
*   **Inclusion of 3-Cells (Tetrahedrons):** To test the upper bounds of cellular lifting, we will extend the complex construction to include 3-cells. The ablation will compare `0-1-2-Cell Complex` versus `0-1-2-3-Cell Complex` configurations to evaluate whether the inclusion of tetrahedrons yields statistically significant gains in expressivity, or merely incurs computational overhead without proportional benefit.
*   **Ricci Flow Iterations:** We will vary the maximum number of Ricci flow iterations $T$ before message passing, mapping the trade-off between topological convergence and computational efficiency.

## 4. Large-Scale Scalability Profiling
While our theoretical complexity bounds the message passing step at $\mathcal{O}(|E| + |F|)$ (where $E$ is the number of edges and $F$ is the number of faces/2-cells), empirical validation is necessary for production viability.

*   **Profiling Methodology:** We will profile the model on massive, multi-million node graphs (e.g., `ogbn-arxiv`, `ogbn-products`).
*   **Metrics:** We will measure peak GPU memory consumption, per-epoch training time, and cellular lifting preprocessing time as a function of graph size.
*   **Benchmarking:** The asymptotic time and space complexity will be plotted against scalable state-of-the-art architectures (e.g., GraphSAGE, GIN). We anticipate demonstrating that while preprocessing incurs an initial cost, the forward-pass complexity remains strictly linear with respect to the lifted complex size.

## 5. Hyperparameter Sensitivity and Statistical Significance
To ensure that performance gains are intrinsic to the architecture rather than the artifact of a lucky random seed or initialization, we will execute a formal sensitivity analysis.

*   **Bayesian Optimization:** We will employ Bayesian Optimization over a defined search space encompassing critical hyperparameters: learning rate, dropout rate, cellular hidden dimension, and Ricci flow step size. 
*   **Statistical Validation:** For the top-performing configurations, we will run $N=10$ independent trials with distinct random seeds. Results will be reported with standard deviations, and a two-sample t-test will be applied against baseline models to confirm that the observed accuracy improvements achieve a significance level of $p < 0.05$. 

This rigorous statistical bounding will guarantee the reproducibility and reliability of the reported results.
