# Neuro-Symbolic IA System Architecture

![Neuro-Symbolic IA System Architecture](/images/Architecture.png "Neuro-Symbolic IA System Architecture")

We present a novel approach based on the integration of Neuro-Symbolic AI systems. The symbolic system is implemented by deductive databases, enhancing the predictive capacity of subsymbolic systems implemented as Knowledge Graph Embedding models. As a proof of concept, we assess the performance of the proposed neuro-symbolic system on top of a KG of lung cancer treatments; the predictive task is to predict treatment effectiveness. The deductive database that specifies the rules relies on our previous work, [“Capturing Knowledge about Drug-Drug Interactions to Enhance Treatment Effectiveness”](https://dl.acm.org/doi/10.1145/3460210.3493560), where we propose a deductive system over knowledge graphs to formalize the process of pharmacokinetic DDIs.

## Running Neuro-Symbolic IA System
- Descriptive_Analysis_Treatment_Response.ipynb: Shows a descriptive analysis of the treatment-response in the KG.
- Analysis_DDIs_by_Treatment_Response.ipynb: Shows a detailed analysis of the distribution and density of DDIs by treatment-response in the benchmarks.
- Statistics_TKG.ipynb: presents the metrics to measure size, diversity, and sparsity in Knowledge Graph.
- Traverse_TKG.ipynb: You can traverse the benchmarks with your onw SPARQL queries.
- deductive_system.py: presents the symbolic system implemented by a deductive database in datalog for the problem of treatment-response.
- Analysis_BoxPlot_CosineSimilarity.ipynb: illustrates the analysis of cosine similarity in a box plot.
- Evaluation_Integrated_SymbolicSubsymbolic_System.ipynb: illustrates the neuro-symbolic system evaluation metrics in the three benchmarks.
- Score_Value_Predicted_Entities.ipynb: shows the behaviour of the scoring function for the entities predicted by each embedding model.
- Link_Prediction.py: executes the link prediction task for the different negative sampling techniques and KGE models.