# Irrigated Cropland Mapping with Geospatial Foundation Models

**Evaluating geospatial foundation models (GeoFMs) for irrigated cropland mapping in North Africa and their
transferability to data-scarce regions**

Irrigated agriculture covers about a fifth of the world's cropland but produces around 40% of its food and accounts for most freshwater withdrawals. Knowing where and how much land is irrigated is therefore essential for water management and food security, especially in water-stressed regions such as North Africa. Yet most countries lack accurate, regularly updated irrigation maps, and existing global products are coarse or static. Geospatial foundation models, large models pretrained on vast archives of satellite imagery, promise accurate maps from few labels and better transfer to new regions. This research evaluates that promise for irrigated cropland mapping, using Morocco as a case study. It asks how accurately these models separate irrigated from rainfed cropland compared with conventional satellite-based methods, how much adapting them to the task helps, how consistent their maps are from year to year, and how well a model trained in Morocco transfers to a region with little ground truth.



<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **Introduction**

    ---

    Why irrigated cropland matters, the research gap, objectives and research questions.

    [:octicons-arrow-right-24: Introduction](introduction.md) ·
    [Proposal (PDF)](assets/research_proposal.pdf)

-   :material-timeline-outline:{ .lg .middle } **Research plan**

    ---

    Five phases: data preparation, model evaluation, transferability, multi-model fusion and annual maps.

    [:octicons-arrow-right-24: Research plan](research_plan.md) · [Phase 1](phase1_data/overview.md) ·
    [Phase 2](phase2_models/overview.md)

-   :material-flask-outline:{ .lg .middle } **Experiments**

    ---

    One setup page and one report per experiment, each reproducible from its configuration.

    [:octicons-arrow-right-24: All experiments](experiments/index.md)

-   :material-chart-bar:{ .lg .middle } **Results**

    ---

    What the experiments have shown so far, and where each research question stands.

    [:octicons-arrow-right-24: Results](results.md) · [Meeting notes](weekly/index.md)

</div>



Setup, data, model weights and run order: [Environment Setup](setup.md).
