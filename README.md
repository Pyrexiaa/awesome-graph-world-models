# Awesome Graph World Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated collection of **papers, datasets, code, and tools** for building **graph world models**: models that keep a structured, graph-shaped state of the world, update it as new observations arrive, and predict how it will change.

The list has a second focus on **fact verification**: treating claims, media, sources, evidence and provenance as an evolving graph, with temporally ordered real/fake news datasets (text, image, video, source and social context) for classification and prediction.

## Table of Contents
1. [Graph World Models](#-graph-world-models)
2. [Dynamic & Temporal Graph Learning](#%EF%B8%8F-dynamic--temporal-graph-learning)
3. [Uncertainty, Evidence Aggregation & Active Acquisition](#-uncertainty-evidence-aggregation--active-acquisition)
4. [Fact Verification & Misinformation Detection Methods](#-fact-verification--misinformation-detection-methods)
5. [Key Datasets for Fact Verification](#-key-datasets-for-fact-verification)
6. [Dataset Comparison Table](#comparison-table)
7. [Graph-Learning Benchmarks (Temporal / Dynamic)](#-graph-learning-benchmarks-temporal--dynamic)
8. [Tools, Libraries & APIs](#%EF%B8%8F-tools-libraries--apis)
9. [Surveys & Further Reading](#-surveys--further-reading)
10. [Practical Notes](#%EF%B8%8F-practical-notes)
11. [References](#-references)

---

## 🌐 Graph World Models

A world model learns a state representation and a **transition function** *P(s<sub>t+1</sub> | s<sub>t</sub>, a<sub>t</sub>)* that can be rolled forward to predict, simulate or plan. In a *graph* world model the state is a graph (entities, relations, attributes) or the transition is computed by message passing. Following the taxonomy of Liu et al. <a href="#ref1">[1]</a>, the table groups methods by what the graph does: **Connector** (spatial / topological memory), **Simulator** (physical and object interactions), **Reasoner** (semantic, knowledge-graph or causal structure), plus **World models *of* graphs**, where the graph itself is the environment whose evolution is predicted.

### Overview Table

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Paper</th>
      <th>Year</th>
      <th>Venue</th>
      <th>Category</th>
      <th>Core Idea</th>
      <th>Data / Environment</th>
      <th>Code</th>
    </tr>
  </thead>
  <tbody>

  <tr>
    <td>Ha &amp; Schmidhuber <a href="#ref2">[2]</a> : World Models</td>
    <td>2018</td>
    <td>arXiv</td>
    <td>Foundation</td>
    <td>VAE + MDN-RNN latent dynamics; agent trained "inside the dream"</td>
    <td>CarRacing, VizDoom</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Battaglia et al. <a href="#ref3">[3]</a> : Interaction Networks</td>
    <td>2016</td>
    <td>NeurIPS</td>
    <td>Simulator (object-centric)</td>
    <td>First learned relational physics simulator over object–relation graphs</td>
    <td>n-body, bouncing balls, strings</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Sanchez-Gonzalez et al. <a href="#ref4">[4]</a> : GN as physics engines</td>
    <td>2018</td>
    <td>ICML</td>
    <td>Simulator</td>
    <td>Graph networks as forward models for inference and model-based control</td>
    <td>Simulated physical systems</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Battaglia et al. <a href="#ref5">[5]</a> : Graph Networks</td>
    <td>2018</td>
    <td>arXiv</td>
    <td>Foundation</td>
    <td>Defines the GN block and <em>relational inductive bias</em> that later GWMs build on</td>
    <td>-</td>
    <td><a href="https://github.com/google-deepmind/graph_nets">Official</a></td>
  </tr>

  <tr>
    <td>Savinov et al. <a href="#ref6">[6]</a> : SPTM</td>
    <td>2018</td>
    <td>ICLR</td>
    <td>Connector</td>
    <td>Semi-parametric topological memory: a graph of observations for navigation</td>
    <td>VizDoom mazes</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Eysenbach et al. <a href="#ref7">[7]</a> : SoRB</td>
    <td>2019</td>
    <td>NeurIPS</td>
    <td>Connector</td>
    <td>Graph search over a replay buffer, with RL distances as edge weights</td>
    <td>Navigation</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Kipf et al. <a href="#ref8">[8]</a> : C-SWM</td>
    <td>2020</td>
    <td>ICLR</td>
    <td>Simulator (object-centric)</td>
    <td>Object slots + action-conditioned GNN transition + contrastive latent loss; multi-step latent rollout scored with Hits@k / MRR</td>
    <td>2D/3D shapes, Atari, 3-body physics</td>
    <td><a href="https://github.com/tkipf/c-swm">Official</a></td>
  </tr>

  <tr>
    <td>Lin et al. <a href="#ref9">[9]</a> : G-SWM</td>
    <td>2020</td>
    <td>ICML</td>
    <td>Simulator (object-centric)</td>
    <td>Generative object-centric world model with interaction and occlusion modelling</td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Sanchez-Gonzalez et al. <a href="#ref10">[10]</a> : GNS</td>
    <td>2020</td>
    <td>ICML</td>
    <td>Simulator (system-centric)</td>
    <td>Encode–process–decode GNN simulator; training-noise injection for long, stable rollouts</td>
    <td>Fluids, sand, goop</td>
    <td><a href="https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate">Official</a></td>
  </tr>

  <tr>
    <td>Pfaff et al. <a href="#ref11">[11]</a> : MeshGraphNets</td>
    <td>2021</td>
    <td>ICLR</td>
    <td>Simulator (system-centric)</td>
    <td>Mesh-space + world-space message passing with adaptive remeshing</td>
    <td>Cloth, CFD, structural</td>
    <td><a href="https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets">Official</a></td>
  </tr>

  <tr>
    <td>Zhang et al. <a href="#ref12">[12]</a> : L3P</td>
    <td>2021</td>
    <td>ICML</td>
    <td>Connector</td>
    <td>"World model as a graph": learned latent landmarks + graph search for planning</td>
    <td>Goal-reaching RL</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Ammanabrolu &amp; Riedl <a href="#ref13">[13]</a> : Worldformer</td>
    <td>2021</td>
    <td>NeurIPS</td>
    <td>Reasoner (KG)</td>
    <td>Predicts knowledge-graph state changes and valid actions in text games</td>
    <td>Jericho text games</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Lam et al. <a href="#ref14">[14]</a> : GraphCast</td>
    <td>2023</td>
    <td>Science</td>
    <td>Simulator (system-centric)</td>
    <td>Multi-mesh GNN autoregressive weather simulator (10-day global forecasts)</td>
    <td>ERA5</td>
    <td><a href="https://github.com/google-deepmind/graphcast">Official</a></td>
  </tr>

  <tr>
    <td>Anokhin et al. <a href="#ref15">[15]</a> : AriGraph</td>
    <td>2025</td>
    <td>IJCAI</td>
    <td>Reasoner (KG + episodic memory)</td>
    <td>LLM agent builds a semantic + episodic memory graph used as a world model</td>
    <td>TextWorld, NetHack</td>
    <td><a href="https://github.com/AIRI-Institute/AriGraph">Official</a></td>
  </tr>

  <tr>
    <td>Rasmussen et al. <a href="#ref16">[16]</a> : Zep / Graphiti</td>
    <td>2025</td>
    <td>arXiv</td>
    <td>Reasoner (temporal KG memory)</td>
    <td>Bi-temporal knowledge graph (event time vs. ingestion time) as agent memory</td>
    <td>DMR, LongMemEval</td>
    <td><a href="https://github.com/getzep/graphiti">Official</a></td>
  </tr>

  <tr>
    <td>Feng et al. <a href="#ref17">[17]</a> : GWM</td>
    <td>2025</td>
    <td>ICML</td>
    <td>Reasoner (multimodal graph)</td>
    <td>Unified graph world model; <em>actions as nodes</em>; token- and embedding-based variants over 6 task families</td>
    <td>Multimodal gen./matching, rec., graph pred., multi-agent, RAG, planning</td>
    <td><a href="https://github.com/ulab-uiuc/GWM">Official</a></td>
  </tr>

  <tr>
    <td>Wang et al. <a href="#ref18">[18]</a> : Dyn-O</td>
    <td>2025</td>
    <td>NeurIPS</td>
    <td>Simulator (object-centric)</td>
    <td>Structured world model built from object-centric representations</td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Feng et al. <a href="#ref19">[19]</a> : FIOC-WM</td>
    <td>2025</td>
    <td>NeurIPS</td>
    <td>Simulator (object-centric)</td>
    <td>Learns object interaction structure for object-centric RL</td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Hu et al. <a href="#ref20">[20]</a> : SGImagineNav</td>
    <td>2025</td>
    <td>arXiv</td>
    <td>Connector (scene graph)</td>
    <td>Imagines unseen parts of a scene graph to guide embodied navigation</td>
    <td>Embodied navigation</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Liu et al. <a href="#ref1">[1]</a> : GWM Survey</td>
    <td>2026</td>
    <td>arXiv</td>
    <td>Survey</td>
    <td>Connector / Simulator / Reasoner taxonomy; open problems: topological plasticity, <em>probabilistic dynamics</em>, hallucinated edges, benchmarking</td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Liu et al. <a href="#ref21">[21]</a> : Graph2Video</td>
    <td>2026</td>
    <td>AAAI</td>
    <td>World model of graphs</td>
    <td>Treats the temporal neighbourhood of a link as a sequence of "graph frames" and models its evolution with video-model machinery</td>
    <td>Reddit, MOOC, Enron, Can. Parl., UCI</td>
    <td><a href="https://github.com/hualiu829/Graph2Video">Official</a></td>
  </tr>

  <tr>
    <td>Nam et al. <a href="#ref22">[22]</a> : Causal-JEPA (C-JEPA)</td>
    <td>2026</td>
    <td>ICML</td>
    <td>Simulator (object-centric, JEPA)</td>
    <td>Object-level latent masking: masked objects must be predicted from the others, giving an interaction-focused inductive bias</td>
    <td>CLEVRER, Push-T</td>
    <td><a href="https://github.com/galilai-group/cjepa">Official</a></td>
  </tr>

  <tr>
    <td>Song &amp; Cai <a href="#ref23">[23]</a> : Rollout error in GWMs</td>
    <td>2026</td>
    <td>arXiv</td>
    <td>World model of graphs (analysis)</td>
    <td>Topology-aware error bounds; Graph Error Amplification Factor ρ(A)·∏‖W‖; spectral regularisation + rollout-consistency training</td>
    <td>Synthetic topologies, agent call-trees, Cora/Citeseer, Bitcoin-Alpha</td>
    <td><a href="https://github.com/Hik289/graph_world_model_accumulative_error">Official</a></td>
  </tr>

  <tr>
    <td>Wang et al. <a href="#ref24">[24]</a> : SD-GWM</td>
    <td>2026</td>
    <td>arXiv</td>
    <td>World model of graphs</td>
    <td>Executable structural contract: node self-dynamics + edge-coupled dynamics (rules, ODEs, domain solvers), a bounded learnable residual and a feasibility projection that enforces hard constraints during rollout</td>
    <td>Semi-synthetic flood testbed, USGS streamflow</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Song &amp; Cai <a href="#ref25">[25]</a> : World-model corrector</td>
    <td>2026</td>
    <td>arXiv</td>
    <td>Reasoner (planning graph)</td>
    <td>Repairs a failed agent planning graph in place, targeting amplification sources (GEAF-guided) instead of replanning the whole graph</td>
    <td>Agent calling-tree testbed, benchmark-inspired topologies</td>
    <td><a href="https://github.com/Hik289/world-model-corrector">Official</a></td>
  </tr>

  <tr>
    <td>Ding et al. <a href="#ref26">[26]</a> : WorldGraph</td>
    <td>2026</td>
    <td>arXiv</td>
    <td>World model of graphs</td>
    <td>Predicts node-, edge- and graph-level transitions; history-aware state encoder, transition controller, transition-aware RL</td>
    <td>trade, genre, reddit, un_vote, contact, socialevo, flights, enron</td>
    <td><a href="https://github.com/USTC-DataDarknessLab/Graph-Native_World_Modeling">Official</a></td>
  </tr>

  <tr>
    <td>Yang et al. <a href="#ref27">[27]</a> : World-as-Graph (WaG)</td>
    <td>2026</td>
    <td>arXiv</td>
    <td>Simulator (object-centric, JEPA)</td>
    <td>Time-varying latent graphs over object slots; relation-aware masking + temporal-GNN memory transition for JEPA-style prediction</td>
    <td>Visual reasoning, robotic manipulation (e.g. PushT)</td>
    <td><a href="https://github.com/Scarlett-Yyq/World-as-Graph">Official</a></td>
  </tr>

  </tbody>
</table>
</div>

### Related (non-graph) world models worth knowing
Used as baselines, design references or definitions: DreamerV3 <a href="#ref28">[28]</a>, V-JEPA 2 <a href="#ref29">[29]</a> (the step from V-JEPA to the action-conditioned V-JEPA 2-AC shows why action conditioning matters), DINO-WM <a href="#ref30">[30]</a>, WebDreamer <a href="#ref31">[31]</a> (an LLM as a world model of the web), R-WoM <a href="#ref32">[32]</a> (retrieval-grounded rollouts for computer-use agents), the world-model survey of Ding et al. <a href="#ref33">[33]</a>, and *Critiques of World Models* <a href="#ref34">[34]</a>. For graph memory in LLM agents, see the survey of Yang et al. <a href="#ref35">[35]</a>.

---

## 🕸️ Dynamic & Temporal Graph Learning

A graph world model needs an encoder for graphs whose nodes, edges and attributes change over time, and a forecasting head. The table covers continuous-time dynamic GNNs, temporal knowledge graph (TKG) forecasting, and the strong-baseline papers that any new method should be compared against.

### Overview Table

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Paper</th>
      <th>Year</th>
      <th>Venue</th>
      <th>Type</th>
      <th>Core Idea</th>
      <th>Datasets</th>
      <th>Code</th>
    </tr>
  </thead>
  <tbody>

  <tr>
    <td>Kumar et al. <a href="#ref36">[36]</a> : JODIE</td>
    <td>2019</td>
    <td>KDD</td>
    <td>CTDG</td>
    <td>Coupled RNNs + projection operator for embedding trajectories</td>
    <td>Reddit, Wikipedia, MOOC, LastFM</td>
    <td><a href="https://github.com/claws-lab/jodie">Official</a></td>
  </tr>

  <tr>
    <td>Hajiramezanali et al. <a href="#ref37">[37]</a> : VGRNN</td>
    <td>2019</td>
    <td>NeurIPS</td>
    <td>Generative DTDG</td>
    <td>Stochastic latent per step gives a <em>distribution</em> over future graphs</td>
    <td>Enron, COLAB, Facebook…</td>
    <td><a href="https://github.com/VGraphRNN/VGRNN">Official</a></td>
  </tr>

  <tr>
    <td>Xu et al. <a href="#ref38">[38]</a> : TGAT</td>
    <td>2020</td>
    <td>ICLR</td>
    <td>CTDG</td>
    <td>Functional time encoding + temporal self-attention</td>
    <td>Reddit, Wikipedia</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Rossi et al. <a href="#ref39">[39]</a> : TGN</td>
    <td>2020</td>
    <td>arXiv</td>
    <td>CTDG</td>
    <td>Per-node memory updated by events; generic framework for continuous-time graphs</td>
    <td>Reddit, Wikipedia, Twitter</td>
    <td><a href="https://github.com/twitter-research/tgn">Official</a></td>
  </tr>

  <tr>
    <td>Wang et al. <a href="#ref40">[40]</a> : CAWN</td>
    <td>2021</td>
    <td>ICLR</td>
    <td>CTDG</td>
    <td>Causal anonymous walks for inductive temporal link prediction</td>
    <td>Reddit, Wikipedia, UCI…</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Jin et al. <a href="#ref41">[41]</a> : RE-NET</td>
    <td>2020</td>
    <td>EMNLP</td>
    <td>TKG forecasting</td>
    <td>Autoregressive event-graph generation over temporal KGs</td>
    <td>ICEWS, GDELT, WIKI, YAGO</td>
    <td><a href="https://github.com/INK-USC/RE-Net">Official</a></td>
  </tr>

  <tr>
    <td>Zhu et al. <a href="#ref42">[42]</a> : CyGNet</td>
    <td>2021</td>
    <td>AAAI</td>
    <td>TKG forecasting</td>
    <td>Copy–generation from historical facts</td>
    <td>ICEWS, GDELT, WIKI, YAGO</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Han et al. <a href="#ref43">[43]</a> : xERTE</td>
    <td>2021</td>
    <td>ICLR</td>
    <td>TKG forecasting</td>
    <td>Explainable temporal subgraph reasoning</td>
    <td>ICEWS</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Li et al. <a href="#ref44">[44]</a> : RE-GCN</td>
    <td>2021</td>
    <td>SIGIR</td>
    <td>TKG forecasting</td>
    <td>Evolutional relational GCN over KG snapshots</td>
    <td>ICEWS, GDELT, WIKI, YAGO</td>
    <td><a href="https://github.com/Lee-zix/RE-GCN">Official</a></td>
  </tr>

  <tr>
    <td>Liu et al. <a href="#ref45">[45]</a> : TLogic</td>
    <td>2022</td>
    <td>AAAI</td>
    <td>TKG forecasting (rules)</td>
    <td>Temporal logical rules mined from random walks; explainable</td>
    <td>ICEWS</td>
    <td><a href="https://github.com/liu-yushan/TLogic">Official</a></td>
  </tr>

  <tr>
    <td>Poursafaei et al. <a href="#ref46">[46]</a> : EdgeBank</td>
    <td>2022</td>
    <td>NeurIPS D&amp;B</td>
    <td>Baseline</td>
    <td>Pure memorisation baseline + harder negative sampling</td>
    <td>Multiple dynamic graphs</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Cong et al. <a href="#ref47">[47]</a> : GraphMixer</td>
    <td>2023</td>
    <td>ICLR</td>
    <td>CTDG</td>
    <td>MLP-Mixer-only temporal model rivals complex architectures</td>
    <td>Reddit, Wikipedia…</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Yu et al. <a href="#ref48">[48]</a> : DyGFormer / DyGLib</td>
    <td>2023</td>
    <td>NeurIPS</td>
    <td>CTDG + library</td>
    <td>Neighbour co-occurrence encoding + patching; unified library of 9+ models</td>
    <td>13 datasets</td>
    <td><a href="https://github.com/yule-BUAA/DyGLib">Official</a></td>
  </tr>

  <tr>
    <td>Liao et al. <a href="#ref49">[49]</a> : GenTKG</td>
    <td>2024</td>
    <td>Findings NAACL</td>
    <td>TKG + LLM</td>
    <td>Temporal-rule retrieval + few-shot instruction tuning of LLMs</td>
    <td>ICEWS, GDELT, YAGO</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Gastinger et al. <a href="#ref50">[50]</a> : Recurrency baseline</td>
    <td>2024</td>
    <td>IJCAI</td>
    <td>Baseline</td>
    <td>"History repeats itself": a simple recurrency baseline beats many TKG models</td>
    <td>ICEWS, GDELT, WIKI, YAGO</td>
    <td><a href="https://github.com/nec-research/recurrency_baseline_tkg">Official</a></td>
  </tr>

  <tr>
    <td>Cornell et al. <a href="#ref51">[51]</a> : Heuristics in temporal graphs</td>
    <td>2025</td>
    <td>arXiv</td>
    <td>Baseline</td>
    <td>Recency/popularity heuristics are competitive on TGB</td>
    <td>TGB</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Gastinger et al. <a href="#ref52">[52]</a> : CountTRuCoLa</td>
    <td>2025</td>
    <td>arXiv</td>
    <td>TKG (rules)</td>
    <td>Rule-confidence learning; strong, explainable TKG forecaster</td>
    <td>TGB 2.0 TKGs</td>
    <td><a href="https://github.com/JuliaGast/counttrucola">Official</a></td>
  </tr>

  <tr>
    <td>Hayes et al. <a href="#ref53">[53]</a> : What do TG models learn?</td>
    <td>2025</td>
    <td>arXiv</td>
    <td>Analysis</td>
    <td>Trained 6,500+ models; they capture preferential attachment but mostly miss recency, density and periodicity</td>
    <td>Synthetic + TGB</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Hu et al. <a href="#ref54">[54]</a> : CFEP</td>
    <td>2026</td>
    <td>Findings ACL</td>
    <td>TKG + conformal</td>
    <td>Coverage-guaranteed prediction sets for future TKG events</td>
    <td>TKG benchmarks</td>
    <td><a href="https://github.com/hucheng-IIE/CFEP">Official</a></td>
  </tr>

  </tbody>
</table>
</div>

> **A warning from the literature.** Simple memorisation (EdgeBank <a href="#ref46">[46]</a>, recurrency <a href="#ref50">[50]</a>, recency/popularity <a href="#ref51">[51]</a>) often matches or beats neural models on ICEWS, GDELT, WIKI, YAGO and TGB. TGB-Seq <a href="#ref55">[55]</a> was built to reduce edge repetition, and reports large gaps between repeated and unseen edges. Include at least one heuristic baseline in every table.

---

## 🎯 Uncertainty, Evidence Aggregation & Active Acquisition

These pieces turn a graph state into a *belief* state: calibrated uncertainty, separating "not enough evidence" (vacuity) from "conflicting evidence" (dissonance), discounting correlated sources, and choosing the next verification action.

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Paper</th>
      <th>Year</th>
      <th>Venue</th>
      <th>Role</th>
      <th>Core Idea</th>
      <th>Code</th>
    </tr>
  </thead>
  <tbody>

  <tr>
    <td>Dong, Berti-Équille &amp; Srivastava <a href="#ref56">[56]</a></td>
    <td>2009</td>
    <td>VLDB</td>
    <td>Source dependence / truth discovery</td>
    <td>Shared <em>false</em> values signal copying; independence-discounted vote counts</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Zhao et al. <a href="#ref57">[57]</a> : S-BGCN / GKDE</td>
    <td>2020</td>
    <td>NeurIPS</td>
    <td>Evidential GNN</td>
    <td>Subjective-logic node uncertainty: vacuity vs. dissonance</td>
    <td><a href="https://github.com/zxj32/uncertainty-GNN">Official</a></td>
  </tr>

  <tr>
    <td>Stadler et al. <a href="#ref58">[58]</a> : Graph Posterior Network</td>
    <td>2021</td>
    <td>NeurIPS</td>
    <td>Evidential GNN</td>
    <td>PPR-diffused Dirichlet evidence for node classification</td>
    <td><a href="https://github.com/stadlmax/Graph-Posterior-Network">Official</a></td>
  </tr>

  <tr>
    <td>Wang et al. <a href="#ref59">[59]</a> : CaGCN</td>
    <td>2021</td>
    <td>NeurIPS</td>
    <td>Calibration</td>
    <td>Topology-aware confidence calibration for GNNs</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Huang et al. <a href="#ref60">[60]</a> : CF-GNN</td>
    <td>2023</td>
    <td>NeurIPS</td>
    <td>Conformal</td>
    <td>Conformal prediction sets for GNNs with coverage guarantees</td>
    <td><a href="https://github.com/snap-stanford/conformalized-gnn">Official</a></td>
  </tr>

  <tr>
    <td>Bickford Smith et al. <a href="#ref61">[61]</a> : EPIG</td>
    <td>2023</td>
    <td>AISTATS</td>
    <td>Active acquisition</td>
    <td>Expected predictive information gain: target the prediction, not the parameters</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Bengs et al. <a href="#ref62">[62]</a></td>
    <td>2024</td>
    <td>ICML</td>
    <td>Critique</td>
    <td>Questions whether evidential DL represents epistemic uncertainty faithfully</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Wang et al. <a href="#ref63">[63]</a> : GNN Uncertainty Survey</td>
    <td>2024</td>
    <td>TMLR</td>
    <td>Survey</td>
    <td>Uncertainty in GNNs; notes that temporal graphs are barely covered</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Wang et al. <a href="#ref64">[64]</a> : NCPNet</td>
    <td>2025</td>
    <td>KDD</td>
    <td>Conformal (temporal)</td>
    <td>Non-exchangeable conformal prediction for temporal GNNs</td>
    <td><a href="https://github.com/ODYSSEYWT/NCPNET">Official</a></td>
  </tr>

  <tr>
    <td>BED-LLM <a href="#ref65">[65]</a></td>
    <td>2025</td>
    <td>arXiv</td>
    <td>Active acquisition</td>
    <td>Bayesian experimental design for LLM information gathering</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Lan et al. <a href="#ref66">[66]</a> : CO-GAT</td>
    <td>2024</td>
    <td>arXiv</td>
    <td>Confidence-aware verification</td>
    <td>Confidence-gated GNN for multi-evidence fact verification</td>
    <td><a href="https://github.com/NEUIR/CO-GAT">Official</a></td>
  </tr>

  <tr>
    <td>INFOGATHERER <a href="#ref67">[67]</a></td>
    <td>2026</td>
    <td>arXiv</td>
    <td>Active acquisition</td>
    <td>Dempster–Shafer belief states; evidence retrieval + strategic questioning</td>
    <td>-</td>
  </tr>

  </tbody>
</table>
</div>

---

## ✅ Fact Verification & Misinformation Detection Methods

**Graph** = builds or reasons over an explicit graph (evidence, propagation, knowledge or concept graph). **Temporal** = uses time explicitly (timestamps, temporal splits, ordered evidence, or time-aware evaluation).

### Overview Table

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Paper</th>
      <th>Year</th>
      <th>Venue</th>
      <th>Graph</th>
      <th>Modalities</th>
      <th>Temporal</th>
      <th>Datasets</th>
      <th>Code</th>
    </tr>
  </thead>
  <tbody>

  <tr>
    <td>Zhou et al. <a href="#ref68">[68]</a> : GEAR</td>
    <td>2019</td>
    <td>ACL</td>
    <td>✅ evidence graph</td>
    <td>Text</td>
    <td>-</td>
    <td>FEVER</td>
    <td><a href="https://github.com/thunlp/GEAR">Official</a></td>
  </tr>

  <tr>
    <td>Liu et al. <a href="#ref69">[69]</a> : KGAT</td>
    <td>2020</td>
    <td>ACL</td>
    <td>✅ kernel GAT</td>
    <td>Text</td>
    <td>-</td>
    <td>FEVER</td>
    <td><a href="https://github.com/thunlp/KernelGAT">Official</a></td>
  </tr>

  <tr>
    <td>Zhong et al. <a href="#ref70">[70]</a> : DREAM</td>
    <td>2020</td>
    <td>ACL</td>
    <td>✅ semantic graph</td>
    <td>Text</td>
    <td>-</td>
    <td>FEVER</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Kim et al. <a href="#ref71">[71]</a> : FactKG</td>
    <td>2023</td>
    <td>ACL</td>
    <td>✅ KG reasoning</td>
    <td>Text + KG (DBpedia)</td>
    <td>-</td>
    <td>FactKG</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Ma et al. <a href="#ref72">[72]</a> : RvNN</td>
    <td>2018</td>
    <td>ACL</td>
    <td>✅ propagation tree</td>
    <td>Text + propagation</td>
    <td>✅ (time-ordered cascades)</td>
    <td>Twitter15/16</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Bian et al. <a href="#ref73">[73]</a> : BiGCN</td>
    <td>2020</td>
    <td>AAAI</td>
    <td>✅ propagation + dispersion</td>
    <td>Text + propagation</td>
    <td>✅</td>
    <td>Weibo, Twitter15/16</td>
    <td><a href="https://github.com/TianBian95/BiGCN">Official</a></td>
  </tr>

  <tr>
    <td>Lu &amp; Li <a href="#ref74">[74]</a> : GCAN</td>
    <td>2020</td>
    <td>ACL</td>
    <td>✅ user graph</td>
    <td>Text + users</td>
    <td>✅ (retweet order)</td>
    <td>Twitter15/16</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Nguyen et al. <a href="#ref75">[75]</a> : FANG</td>
    <td>2020</td>
    <td>CIKM</td>
    <td>✅ heterogeneous social graph</td>
    <td>Text + users + sources</td>
    <td>✅ (engagement time)</td>
    <td>Twitter (news-source)</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Wei et al. <a href="#ref76">[76]</a> : EBGCN</td>
    <td>2021</td>
    <td>ACL</td>
    <td>✅ Bayesian edges</td>
    <td>Text + propagation</td>
    <td>✅</td>
    <td>PHEME, Twitter15/16</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Dou et al. <a href="#ref77">[77]</a> : UPFD</td>
    <td>2021</td>
    <td>SIGIR</td>
    <td>✅ propagation graph</td>
    <td>Text + user history</td>
    <td>-</td>
    <td>UPFD (PolitiFact, GossipCop)</td>
    <td><a href="https://github.com/safe-graph/GNN-FakeNews">Official</a></td>
  </tr>

  <tr>
    <td>Yang et al. <a href="#ref78">[78]</a> : CofCED</td>
    <td>2022</td>
    <td>COLING</td>
    <td>-</td>
    <td>Text + raw reports</td>
    <td>-</td>
    <td>LIAR-RAW, RAWFC</td>
    <td><a href="https://github.com/Nicozwy/CofCED">Official</a></td>
  </tr>

  <tr>
    <td>Allein et al. <a href="#ref79">[79]</a> : Time-aware evidence ranking</td>
    <td>2020</td>
    <td>arXiv</td>
    <td>-</td>
    <td>Text</td>
    <td>✅ (evidence recency)</td>
    <td>MultiFC</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Shao et al. <a href="#ref80">[80]</a> : HAMMER</td>
    <td>2023</td>
    <td>CVPR</td>
    <td>-</td>
    <td>Image + text</td>
    <td>-</td>
    <td>DGM4</td>
    <td><a href="https://github.com/rshaojimmy/MultiModal-DeepFake">Official</a></td>
  </tr>

  <tr>
    <td>Qi et al. <a href="#ref81">[81]</a> : SV-FEND</td>
    <td>2023</td>
    <td>AAAI</td>
    <td>-</td>
    <td>Video + audio + text + comments</td>
    <td>✅ (temporal split)</td>
    <td>FakeSV</td>
    <td><a href="https://github.com/ICTMCG/FakeSV">Official</a></td>
  </tr>

  <tr>
    <td>Qi et al. <a href="#ref82">[82]</a> : SNIFFER</td>
    <td>2024</td>
    <td>CVPR</td>
    <td>-</td>
    <td>Image + text (MLLM)</td>
    <td>-</td>
    <td>NewsCLIPpings</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Bu et al. <a href="#ref83">[83]</a> : FakingRecipe</td>
    <td>2024</td>
    <td>ACM MM</td>
    <td>-</td>
    <td>Video (creative process)</td>
    <td>~</td>
    <td>FakeSV, FakeTT</td>
    <td><a href="https://github.com/ICTMCG/FakingRecipe">Official</a></td>
  </tr>

  <tr>
    <td>Kim et al. <a href="#ref84">[84]</a> : DAWN</td>
    <td>2025</td>
    <td>WSDM</td>
    <td>✅ graph structure learning</td>
    <td>Text + engagements</td>
    <td>✅ (temporality-aware eval.: train only on engagements before the cut-off; early engagements down-weight noisy edges)</td>
    <td>PolitiFact, GossipCop</td>
    <td><a href="https://github.com/LeeJunmo/DAWN">Official</a></td>
  </tr>

  <tr>
    <td>Liu et al. <a href="#ref85">[85]</a> : MMD-Agent</td>
    <td>2025</td>
    <td>ICLR</td>
    <td>-</td>
    <td>Image + text (LVLM agent)</td>
    <td>-</td>
    <td>MMFakeBench</td>
    <td><a href="https://github.com/liuxuannan/MMFakeBench">Official</a></td>
  </tr>

  <tr>
    <td>Barik et al. <a href="#ref86">[86]</a> : ChronoFact</td>
    <td>2025</td>
    <td>IJCAI</td>
    <td>- (event timelines)</td>
    <td>Text</td>
    <td>✅ (event order)</td>
    <td>ChronoClaims, T-FEVER, T-FEVEROUS, T-QuanTemp</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Braun et al. <a href="#ref87">[87]</a> : DEFAME</td>
    <td>2025</td>
    <td>ICML</td>
    <td>-</td>
    <td>Image + text (agentic, tools)</td>
    <td>~ (dated search)</td>
    <td>AVeriTeC, MOCHEG, VERITE</td>
    <td><a href="https://github.com/multimodal-ai-lab/DEFAME">Official</a></td>
  </tr>

  <tr>
    <td>Fu et al. <a href="#ref88">[88]</a> : LiveVQA</td>
    <td>2025</td>
    <td>NeurIPS</td>
    <td>-</td>
    <td>Image + text (live knowledge)</td>
    <td>✅ (post-cutoff)</td>
    <td>LiveVQA</td>
    <td><a href="https://github.com/fumingyang2004/LIVEVQA">Official</a></td>
  </tr>

  <tr>
    <td>Hu et al. <a href="#ref89">[89]</a> : DTN</td>
    <td>2025</td>
    <td>PeerJ CS</td>
    <td>✅ heterogeneous user–news–post graph</td>
    <td>Text + image + social</td>
    <td>✅ (time-similarity weighting)</td>
    <td>PHEME, GossipCop</td>
    <td>-</td>
  </tr>

  <tr>
    <td>EGMMG <a href="#ref90">[90]</a></td>
    <td>2025</td>
    <td>arXiv</td>
    <td>✅ evidence graph (attention GNN)</td>
    <td>Image + text</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <td>MEVER <a href="#ref91">[91]</a></td>
    <td>2026</td>
    <td>EACL</td>
    <td>✅ two-layer multimodal graph</td>
    <td>Image + text</td>
    <td>-</td>
    <td>Multimodal claim verification</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Zhang et al. <a href="#ref92">[92]</a> : IEEG / VMD-FACT</td>
    <td>2026</td>
    <td>CVPR</td>
    <td>✅ multimodal evidence DAG (evidence + fact-check results + dependencies), distilled into a 7B MLLM (VideoLLaMA2-7B baseline)</td>
    <td>Video + audio + text</td>
    <td>-</td>
    <td>RAVM (+ transfer to FakeSV, FakeTT, FMNV)</td>
    <td>-</td>
  </tr>

  <tr>
    <td>Yang et al. <a href="#ref93">[93]</a> : PCGR</td>
    <td>2026</td>
    <td>CVPR</td>
    <td>✅ probabilistic concept DAG</td>
    <td>Image + text</td>
    <td>-</td>
    <td>-</td>
    <td><a href="https://github.com/2302Jerry/pcgr">Official</a></td>
  </tr>

  <tr>
    <td>Wang et al. <a href="#ref94">[94]</a> : RETSIMD</td>
    <td>2026</td>
    <td>AAAI</td>
    <td>-</td>
    <td>Image + text</td>
    <td>-</td>
    <td>-</td>
    <td><a href="https://github.com/wangbing1416/RETSIMD">Official</a></td>
  </tr>

  <tr>
    <td>Lin et al. <a href="#ref95">[95]</a> : EvoGraph-R1</td>
    <td>2026</td>
    <td>CVPR</td>
    <td>✅ self-evolving hypergraph</td>
    <td>Image + text (QA, agentic RL)</td>
    <td>-</td>
    <td>Text QA, E-VQA, multimodal QA</td>
    <td><a href="https://github.com/ninjaX2o/EvoGraph-R1">Official</a></td>
  </tr>

  <tr>
    <td>Papadopoulos et al. <a href="#ref96">[96]</a> : TRENT</td>
    <td>2026</td>
    <td>ECCV</td>
    <td>- (relational fusion)</td>
    <td>Image + text + evidence</td>
    <td>✅ (2017–2025 Community Notes)</td>
    <td>X-POSE</td>
    <td><a href="https://github.com/stevejpapad/evidence-triangulation">Official</a></td>
  </tr>

  </tbody>
</table>
</div>

**Evaluation-methodology papers to cite.** *Novel Claim or Déjà Vu?* <a href="#ref97">[97]</a> finds that 17–29% of supposedly post-cutoff multimodal claims may still be contaminated. The AVerImaTeC shared-task overview <a href="#ref98">[98]</a> gives current state-of-the-art numbers on real image–text claims. DAWN <a href="#ref84">[84]</a> shows that standard fake-news evaluation leaks future engagements.

---

## 📂 Key Datasets for Fact Verification

Grouped by what each dataset gives a graph builder. Every entry lists its **temporal signal**, since that decides whether the data supports dynamic or longitudinal modelling.

### A. Dynamic & Time-Aware Benchmarks (verdicts or splits that depend on time)

#### VeriTaS (2026) <a href="#ref99">[99]</a>
- Introduction: The first <em>dynamic</em> multimodal fact-checking benchmark (TU Darmstadt). Version 2 (Apr 2026) has 25K claims from 104 fact-checking organisations in 54 languages, with 8,692 images and 5,334 videos. Claims come mostly from Facebook (38.4%) and X/Twitter (20.5%); AFP supplies 24.8% of reviews. Instead of one label it scores four properties on a continuous [−1, 1] scale, where 0 = NEI: <strong>media authenticity, media contextualisation, veracity, context coverage</strong> (cherry-picking). A fifth score, <strong>integrity</strong>, is the worst of properties 2–4, so a claim is <em>Intact</em> or <em>Compromised</em>. Verdicts come from an ensemble of four LLMs with justifications, checked by human evaluation. <em>Caveat:</em> only ~0.3% of fact-checked claims are Intact, so most Intact claims are LLM-<strong>rectified</strong> (corrected) versions of false claims. v1 (Jan 2026) reported ~24K claims, 8,377 images, 4,785 videos and 108 organisations, so cite the version you use.
- Temporal signal: 25 <strong>quarterly splits</strong> from Q1 2020 to Q1 2026 (1K claims each, balanced Intact/Compromised), plus a longitudinal split of 2,500 claims (100 per quarter). New claims are added each quarter, and the authors commit to continuing until at least Q4 2028. They note that only the most recent quarter is reliable for benchmarking, because of contamination. Access is through a gated researcher request.
- Link: [Website](https://veritas.mai.informatik.tu-darmstadt.de/) | [Paper](https://aclanthology.org/2026.acl-long.1948/)

#### LiveFact (2026) <a href="#ref100">[100]</a>
- Introduction: A fake-news benchmark refreshed <strong>monthly</strong> that simulates the "fog of war". The November 2025 release covers 737 news events with 25,064 evidence items retrieved through Google APIs, bounded to three time slices per event (3 days before, on, and 3 days after the headline date). It contains 4,392 claims (1,468 Real / 1,451 Fake / 1,473 Ambiguous) that o4-mini <strong>generates</strong> from the headline and dated evidence, so these are synthetic claims about real events. There are two modes, classification and inference, and an SSA factor that measures contamination. 22 LLMs were evaluated.
- Temporal signal: In inference mode the gold label depends on the evidence slice: at −3 days 3,698 of 4,392 claims (84.2%) are <em>Ambiguous</em>, falling to 1,598 at day 0 and 1,489 at +3 days. This is the closest existing setup to belief trajectories, but it is text-only.
- Link: [Paper](https://aclanthology.org/2026.acl-long.546/) | [Code](https://github.com/bebxy/livefact)

#### CommunityFact (2026) <a href="#ref101">[101]</a>
- Introduction: 15,992 standalone claims built from X Community Notes (9,970 True / 6,022 False), in 5 languages (EN, ES, FR, JA, PT) and 2 domains (politics, finance). Includes the evidence URLs cited by note writers. Can be regenerated from future Community Notes archive snapshots. CC BY 4.0.
- Temporal signal: Timestamped notes and temporal train/test splits (notes from 2025, archive to 27 March 2026). Text-only.
- Link: [Code](https://github.com/sahajps/CommunityFact) | [Hugging Face](https://huggingface.co/datasets/sahajps/CommunityFact)

#### ChronoClaims (2025) <a href="#ref86">[86]</a>
- Introduction: 40,249 / 3,544 / 3,735 train / val / test claims built from a Wikidata (Nov 2022) snapshot. Each claim has 3–5 events with explicit or implicit time expressions, overlapping and recurring events. Labels: SUPPORT / REFUTE. Also evaluated: temporal subsets T-FEVER, T-FEVEROUS and T-QuanTemp.
- Temporal signal: The task is to check event order and temporal consistency against evidence timelines.
- Link: [Paper](https://www.ijcai.org/proceedings/2025/0893.pdf) (no public link found)

#### QuanTemp (2024) <a href="#ref102">[102]</a>
- Introduction: Real-world <strong>numerical</strong> claims from fact-checkers, typed as statistical, <strong>temporal</strong>, comparison or interval, with a released evidence corpus and BM25 rankings. CC BY-NC 4.0.
- Temporal signal: Has an explicit temporal-claim category; claims carry fact-check dates.
- Link: [Code / Data](https://github.com/factiverse/QuanTemp)

#### TSVer (2025) <a href="#ref103">[103]</a>
- Introduction: 304 real-world claims from 41 fact-checking organisations, verified against a curated database of 400 <strong>time series</strong>; κ = 0.77 on verdicts.
- Temporal signal: The evidence itself is temporal: claims about trends, levels and changes over time.
- Link: [Paper](https://aclanthology.org/2025.emnlp-main.1519/)

#### M4FC (2025) <a href="#ref104">[104]</a>
- Introduction: 4,982 images and 6,980 claims from 22 IFCN fact-checkers in 17 countries, 10 languages. Six tasks: visual claim extraction, claimant intent, fake-image detection, <strong>image contextualisation</strong>, <strong>location verification</strong> and verdict prediction. CC BY-SA 4.0.
- Temporal signal: Train / dev / test split by fact-check date (a temporal split).
- Link: [Code / Data](https://github.com/UKPLab/M4FC)

#### MMM-Fact (2025) <a href="#ref105">[105]</a>
- Introduction: 125,449 fact-checked claims from <strong>1995–2025</strong> (four fact-checking sites plus one news outlet). Each claim has its full fact-check article and evidence in text, image, video and table form. Labels: True / False / NEI. Evidence difficulty is tiered by source count: Basic (1–5), Intermediate (6–10), Advanced (>10).
- Temporal signal: 30-year span, so it supports longitudinal analysis; evidence links are dated.
- Link: [Hugging Face](https://huggingface.co/datasets/Wenyan0110/MMM-Fact)

### B. Image–Text & Multimodal Fact-Checking (real-world claims with evidence)

#### AVerImaTeC (2025) <a href="#ref106">[106]</a>
- Introduction: 1,297 real-world image–text claims with question–answer evidence (answers cite URLs and medium: web text, video, PDF, image) and textual justifications. Labels: Supported / Refuted / Not Enough Evidence / Conflicting Evidence–Cherry-picking. Metadata includes speaker, publisher, <strong>publication date</strong> and location. CC BY-NC 4.0. Used in the FEVER 2026 shared task.
- Temporal signal: Evidence is restricted to material published before the claim date.
- Link: [Hugging Face](https://huggingface.co/datasets/Rui4416/AVerImaTeC) | [Website](https://fever.ai/dataset/averimatec.html)

#### 5Pils (2024) <a href="#ref107">[107]</a>
- Introduction: 1,676 fact-checked images annotated with the "5 Pillars" of image meta-context: <strong>provenance/source, date, location, motivation</strong> and image type (manipulated or not). CC BY-SA 4.0.
- Temporal signal: Recovering the original date of the image is one of the five targets.
- Link: [Code / Data](https://github.com/UKPLab/5pils)

#### COVE (2025) <a href="#ref108">[108]</a>
- Introduction: Context-and-veracity prediction for out-of-context images. Predicts the original context of an image (seven context items) and then the veracity of the claim.
- Temporal signal: Context includes the image's original date.
- Link: [Code](https://github.com/UKPLab/naacl2025-cove)

#### MOCHEG (2023) <a href="#ref109">[109]</a>
- Introduction: Real claims from PolitiFact and Snopes, with text and image evidence and ruling explanations, for end-to-end multimodal fact-checking (retrieval, verdict, explanation).
- Temporal signal: Claims are dated through their source fact-check articles.
- Link: [Code / Data](https://github.com/VT-NLP/Mocheg)

#### VERITE (2024) <a href="#ref110">[110]</a>
- Introduction: Real-world benchmark built to remove <em>unimodal bias</em>. Balanced true, out-of-context and miscaptioned image–text pairs drawn from fact-checks.
- Temporal signal: -
- Link: [Code / Data](https://github.com/stevejpapad/image-text-verification)

#### X-POSE (2026) <a href="#ref96">[96]</a>
- Introduction: 5,704 unique image–text pairs (2,881 truthful / 2,823 misinformation) labelled through <strong>X Community Notes</strong>, spanning <strong>2017–2025</strong>, with retrieved external evidence.
- Temporal signal: Eight years of dated posts.
- Link: [Code / Data](https://github.com/stevejpapad/evidence-triangulation)

#### MultiClaim (2023) <a href="#ref111">[111]</a>
- Introduction: 205,751 fact-checks in 39 languages and 28,092 social-media posts in 27 languages, linked by 31,305 post–fact-check pairs. Posts include text, <strong>OCR of attached images</strong>, publication date, platform and rating. Machine-translated English versions are provided. Available on request (research use).
- Temporal signal: Post publication dates are available for about 26K posts.
- Link: [Zenodo](https://zenodo.org/records/7737983) | [Code](https://github.com/kinit-sk/multiclaim)

#### Factify 2 (2023) <a href="#ref112">[112]</a>
- Introduction: Multimodal claim–document pairs (text and images) with 5 entailment-style classes, including satire. Used in the De-Factify shared task at AAAI 2023.
- Temporal signal: -
- Link: [Paper](https://arxiv.org/abs/2304.03897)

#### MR2 (2023) <a href="#ref113">[113]</a>
- Introduction: 14,700 real-world English (Twitter) and Chinese (Weibo) posts with images. Each post comes with retrieved <strong>image and text evidence</strong> plus metadata and comments. Fact-checker verdicts are mapped to 3 classes: Rumor / Non-Rumor / Unverified.
- Temporal signal: -
- Link: [Code / Data](https://github.com/THU-BPM/MR2) | [Paper](https://dl.acm.org/doi/10.1145/3539618.3591896)

### C. Synthetic / Out-of-Context Multimodal Misinformation

#### NewsCLIPpings (2021) <a href="#ref114">[114]</a>
- Introduction: Large-scale, automatically generated out-of-context image–caption pairs (built on VisualNews) using CLIP, SBERT, person- and scene-matching strategies.
- Temporal signal: Inherits article dates from VisualNews.
- Link: [Code / Data](https://github.com/g-luo/news_clippings)

#### COSMOS (2023) <a href="#ref115">[115]</a>
- Introduction: 200K+ images with 450K+ captions for self-supervised training, plus a manually labelled out-of-context test set.
- Temporal signal: -
- Link: [Code / Data](https://github.com/shivangi-aneja/COSMOS)

#### DGM4 (2023) <a href="#ref80">[80]</a>
- Introduction: ~230K news image–text pairs with <strong>grounded</strong> manipulations: face swap or attribute edit in the image, text swap or sentiment edit in the caption. Manipulated regions and tokens are annotated.
- Temporal signal: -
- Link: [Code / Data](https://github.com/rshaojimmy/MultiModal-DeepFake)

#### MMFakeBench (2025) <a href="#ref85">[85]</a>
- Introduction: 11,000 image–text pairs covering 3 critical sources of misinformation (textual veracity distortion, visual veracity distortion, cross-modal consistency distortion) and 12 forgery subtypes, including AI-generated images. Built to evaluate LVLMs.
- Temporal signal: -
- Link: [Code / Data](https://github.com/liuxuannan/MMFakeBench)

#### Twitter-COMMs (2022) <a href="#ref116">[116]</a>
- Introduction: Large set of tweets with images on climate, COVID and military topics, with automatically generated out-of-context mismatches and an evaluation set.
- Temporal signal: Tweets are timestamped.
- Link: [Paper](https://aclanthology.org/2022.naacl-main.110/)

#### XFacta (2025) <a href="#ref117">[117]</a>
- Introduction: Contemporary real-world image–text misinformation built by an automatic pipeline around current trending topics, to avoid stale events. Released in batches.
- Temporal signal: Built for recency; the pipeline can be re-run on new events.
- Link: [Code / Data](https://github.com/neu-vi/XFacta)

### D. News Articles + Sources + Social Context (graph-ready)

#### FakeNewsNet (2020) <a href="#ref118">[118]</a>
- Introduction: PolitiFact (432 fake / 624 real) and GossipCop (5,323 fake / 16,817 real) articles with text, images and social context: tweets, retweets, replies, likes, user profiles, follower/followee networks.
- Temporal signal: <strong>Spatiotemporal</strong>: timestamps for news and every engagement, plus user locations. Supports propagation and early-detection studies.
- Link: [GitHub](https://github.com/KaiDMML/FakeNewsNet) (social context must be rehydrated through the X/Twitter API)

#### UPFD (2021) <a href="#ref77">[77]</a>
- Introduction: Ready-made propagation graphs from FakeNewsNet: news root plus retweeting users, with user-history features (BERT/spaCy/profile). Loads in one line from PyTorch Geometric.
- Temporal signal: Propagation order is preserved in the graph structure.
- Link: [PyG dataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.UPFD.html) | [Code](https://github.com/safe-graph/GNN-FakeNews)

#### MuMiN (2022) <a href="#ref119">[119]</a>
- Introduction: A <strong>heterogeneous graph</strong> of 12,914 fact-checked claims (from 115 fact-checkers, 41 languages) linked to 21.5M tweets, 26,048 threads and ~2M users, plus images and articles. Tasks: claim and tweet classification (misinformation / factual).
- Temporal signal: Spans about a decade; nodes and edges are timestamped.
- Link: [Website](https://mumin-dataset.github.io/) | [Builder](https://github.com/MuMiN-dataset/mumin-build) (needs an X/Twitter API key to rehydrate)

#### NELA-GT (2018–2022) <a href="#ref120">[120]</a>
- Introduction: Yearly corpora of news articles with <strong>source-level</strong> reliability labels (reliable / mixed / unreliable, aggregated from MBFC and others). NELA-GT-2022: 1,778,361 articles from 361 sources; 2021: 1,856,509 / 367; 2020: 1,779,127 / 519, plus COVID-19 and election subsets.
- Temporal signal: Every article has a publication date. Ideal for source nodes, cross-source copying and drift studies.
- Link: [GitHub](https://github.com/MELALab/nela-gt) (Harvard Dataverse)

#### FineFake (2024/2026) <a href="#ref121">[121]</a>
- Introduction: Multi-domain (politics, entertainment, business, health, society, conflict), multi-platform (Snopes, Twitter, Reddit, CNN, AP News, CDC, NYT, Washington Post) text + image news. <strong>Knowledge-enriched</strong> with Wikidata entities, descriptions, relations and KG embeddings. Six fine-grained labels: real, text–image inconsistency, content–knowledge inconsistency, text-based fake, image-based fake, others.
- Temporal signal: Publication date for each item.
- Link: [GitHub](https://github.com/Accuser907/FineFake)

#### CoAID / ReCOVery / MM-COVID (2020) <a href="#ref122">[122]</a> <a href="#ref123">[123]</a> <a href="#ref124">[124]</a>
- Introduction: COVID-19 misinformation repositories with news articles, social posts and user engagements. ReCOVery adds source-credibility and multimodal content; MM-COVID is multilingual and multimodal.
- Temporal signal: Engagements are timestamped (early-2020 to late-2020 windows).
- Link: [CoAID](https://github.com/cuilimeng/CoAID) | [ReCOVery](https://github.com/apurvamulay/ReCOVery) | [MM-COVID paper](https://arxiv.org/abs/2011.04088)

#### Weibo21 / MCFEND (2021 / 2024) <a href="#ref125">[125]</a> <a href="#ref126">[126]</a>
- Introduction: Chinese fake-news datasets. Weibo21 covers 9 domains for multi-domain detection. MCFEND has 23,789 news pieces verified by 14 fact-checking agencies and drawn from many platforms (Weibo, WeChat, Douyin, Toutiao, Zhihu, news outlets), with social context: posts, about 2.1M comments, user profiles. A detector trained on Weibo-21 drops from F1 0.943 to 0.470 when tested on MCFEND, which is a strong argument for multi-source evaluation.
- Temporal signal: Post timestamps.
- Link: [Weibo21](https://github.com/kennqiang/MDFEND-Weibo21) | [MCFEND](https://github.com/TrustworthyComp/MCFEND)

### E. Rumour Propagation Cascades (classic temporal graphs)

#### Twitter15 / Twitter16 (2017) <a href="#ref127">[127]</a>
- Introduction: 1,490 / 818 source tweets, each with its retweet/reply <strong>propagation tree</strong>. 4 classes: non-rumour, false, true, unverified.
- Temporal signal: Each node in a tree carries a time delay from the source tweet.
- Link: [Paper](https://aclanthology.org/P17-1066/) (widely mirrored. Read the caveat in [Practical Notes](#%EF%B8%8F-practical-notes) first)

#### PHEME (2016) <a href="#ref128">[128]</a>
- Introduction: Conversation threads around 9 breaking-news events (6,425 threads in the commonly used rumour / non-rumour version) with veracity annotations.
- Temporal signal: Threads are timestamped; supports event-based (leave-one-event-out) splits.
- Link: [Datasets](https://www.zubiaga.org/datasets/)

#### Weibo rumour (2016) <a href="#ref129">[129]</a>
- Introduction: Chinese rumour and non-rumour events from Sina Weibo, each with its repost sequence.
- Temporal signal: Repost timestamps.
- Link: [Paper](https://www.ijcai.org/Proceedings/16/Papers/537.pdf)

### F. Image / Social-Media Multimodal Fake-News Classification

#### Fakeddit (2020) <a href="#ref130">[130]</a>
- Introduction: About 1M Reddit submissions from many subreddits, with 2-, 3- and 6-way labels. Most have images; comments and submission metadata are included.
- Temporal signal: Submission timestamps (2008–2019).
- Link: [GitHub](https://github.com/entitize/Fakeddit)

#### Weibo (multimodal) & MediaEval Image Verification Corpus (2016–2017) <a href="#ref131">[131]</a> <a href="#ref132">[132]</a>
- Introduction: The two standard image–text social-media benchmarks used by most multimodal fake-news detectors (EANN, SAFE, etc.).
- Temporal signal: Post timestamps; MediaEval is split by event.
- Link: [Image Verification Corpus](https://github.com/MKLab-ITI/image-verification-corpus)

### G. Video & Audio

#### FakeSV (2023) <a href="#ref81">[81]</a>
- Introduction: The largest Chinese short-video fake-news dataset (Douyin / Kuaishou): 1,827 fake, 1,827 real and 1,884 debunking videos. Includes title, video, audio, comments, user profiles and publish time. Ships with <strong>event-based and temporal splits</strong>. Access requires a data-use agreement.
- Temporal signal: Publish time; temporal split. Of 434 events with debunking videos, 39% had fake videos posted <em>after</em> the debunk, so debunking does not end a claim's life cycle.
- Link: [GitHub](https://github.com/ICTMCG/FakeSV)

#### FakeTT (2024) <a href="#ref83">[83]</a>
- Introduction: English TikTok fake-news videos: 1,172 fake and 819 real, covering 286 events, built from Snopes reports published Jan 2018–Jan 2024. Access by application form.
- Temporal signal: Six-year span; temporal split.
- Link: [GitHub](https://github.com/ICTMCG/FakingRecipe)

#### FMNV (2025) <a href="#ref133">[133]</a>
- Introduction: 2,393 <strong>media-published</strong> news videos (YouTube and Twitter, 27 outlets, 12 topics, 5 years): 893 real and 1,500 fake in four types (contextual dishonesty, cherry-picked editing, synthetic voice-over, contrived absurdity).
- Temporal signal: Five-year span.
- Link: [GitHub](https://github.com/DennisIW/FMNV)

#### RAVM (VMD-FACT, 2026) <a href="#ref92">[92]</a>
- Introduction: <strong>Realistic AI-generated video misinformation</strong>: 9,049 claim–video pairs (4,355 real / 4,694 fake), split 6,028 / 1,500 / 1,521 train / val / test. A multi-agent framework starts from trending events and manipulates four sources while keeping the modalities consistent. <em>Claim</em> manipulation: replacement, rewrite, generation, guided by intent polarity. <em>Video</em> manipulation: keyframe-to-video fabrication and regeneration with open video generators (e.g. LTX-Video, EasyAnimate). <em>Audio</em> manipulation: background music and synthesised speech. <em>Cross-modal</em> manipulation. Some samples are drawn from FakeTT and FMNV. Each pair is annotated for veracity, intent polarity and attribution. Gemini 2.5 reaches only 68.89% accuracy; the paper's 7B IEEG model reaches 75.99%. Fine-tuning on RAVM raises accuracy on FakeSV, FakeTT and FMNV by 13.48, 36.65 and 15.37 points.
- Temporal signal: -
- Link: [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_VMD-FACT_A_New_Video_Dataset_and_MLLM-based_method_for_Detecting_CVPR_2026_paper.html) | Dataset link given in the paper: `https://gitee.com/VR_NAVE/ravm` (<strong>not reachable as of Oct 2026</strong>; no mirror found. Contact the corresponding authors, Dongyu She or Zhong Zhou.)

#### Fake Video Corpus (FVC, 2019) <a href="#ref134">[134]</a>
- Introduction: Debunked and verified user-generated videos with their <strong>near-duplicate cascades</strong> (re-uploads across YouTube, Facebook and Twitter): 2,916 fake and 2,090 real videos including near-duplicates, in English, French, Russian, German and Arabic (counts as reported in the FakeSV comparison table).
- Temporal signal: Re-uploads are dated, so a claim's video can be followed as it reappears over months.
- Link: [GitHub](https://github.com/MKLab-ITI/fake-video-corpus)

#### COVID-VTS (2023) <a href="#ref135">[135]</a>
- Introduction: Fact extraction and verification on short COVID-19 videos from Twitter.
- Temporal signal: Tweet timestamps.
- Link: [GitHub](https://github.com/FuxiaoLiu/Twitter-Video-dataset)

### H. Text Claim Verification (classic baselines)

#### FEVER (2018) / AVeriTeC (2023) <a href="#ref136">[136]</a> <a href="#ref137">[137]</a>
- Introduction: FEVER has 185K Wikipedia-based claims (Supported / Refuted / NEI). AVeriTeC has 4,568 real claims from 50 fact-checkers with QA-pair evidence retrieved from the web; it adds Conflicting Evidence / Cherry-picking.
- Temporal signal: AVeriTeC restricts evidence to material published before the claim date (no temporal leakage).
- Link: [FEVER](https://fever.ai/dataset/fever.html) | [AVeriTeC](https://fever.ai/dataset/averitec.html)

#### LIAR (2017) / LIAR-RAW & RAWFC (2022) <a href="#ref138">[138]</a> <a href="#ref78">[78]</a>
- Introduction: LIAR has 12.8K short PolitiFact statements with 6 labels and speaker metadata (party, job, state, context, credit history). LIAR-RAW and RAWFC add the raw reports used as evidence.
- Temporal signal: Statement dates are available through PolitiFact.
- Link: [LIAR paper](https://arxiv.org/abs/1705.00648) | [LIAR-RAW / RAWFC](https://github.com/Nicozwy/CofCED/tree/main/Datasets)

#### MultiFC (2019) / X-Fact (2021) / Snopes corpus (2019) <a href="#ref139">[139]</a> <a href="#ref140">[140]</a> <a href="#ref141">[141]</a>
- Introduction: MultiFC has 34,918 claims from 26 fact-checking sites with evidence pages and rich metadata. X-Fact has 31,189 claims in 25 languages. The Snopes corpus annotates evidence and stance.
- Temporal signal: Claim dates are present in the metadata, which made time-aware evidence ranking possible <a href="#ref79">[79]</a>.
- Link: [MultiFC](https://arxiv.org/abs/1909.03242) | [X-Fact](https://github.com/utahnlp/x-fact)

### I. Source-Reliability Labels (priors for SOURCE nodes)

#### Media factuality & domain-quality ratings <a href="#ref142">[142]</a> <a href="#ref143">[143]</a>
- Introduction: Baly et al. give factuality and bias labels for about 1,000 news media, derived from MBFC. Lin et al. aggregate several expert rating sets into a single quality score for ~11K news domains.
- Temporal signal: Ratings are snapshots, so record the snapshot date when you join them to articles.
- Link: [News-Media-Reliability](https://github.com/ramybaly/News-Media-Reliability) | [domain-quality-ratings](https://github.com/hauselin/domain-quality-ratings)


---

### Comparison Table

Legend: **T** text, **I** image, **V** video, **A** audio, **S** social context (users, engagements, propagation), **Src** source/publisher labels or metadata, **E** retrieved evidence. ✅ = real timestamps usable for temporal modelling; 🔁 = dynamic, periodically refreshed benchmark.

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Language</th>
      <th>Modalities</th>
      <th>Temporal</th>
      <th>Labels</th>
      <th>Task</th>
      <th>Access</th>
    </tr>
  </thead>
  <tbody>

  <tr>
    <td>VeriTaS <a href="#ref99">[99]</a></td>
    <td>54 languages</td>
    <td>T, I, V, Src</td>
    <td>✅ 🔁 25 quarterly splits (2020–2026)</td>
    <td>4 properties + integrity on [−1, 1]</td>
    <td>Decomposed verification</td>
    <td>Gated request</td>
  </tr>

  <tr>
    <td>LiveFact <a href="#ref100">[100]</a></td>
    <td>English</td>
    <td>T, E</td>
    <td>✅ 🔁 monthly; −3/0/+3-day evidence</td>
    <td>Real / Fake / Ambiguous (slice-dependent)</td>
    <td>Time-aware fake news (LLM-generated claims)</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>CommunityFact <a href="#ref101">[101]</a></td>
    <td>EN, ES, FR, JA, PT</td>
    <td>T, E (note URLs)</td>
    <td>✅ 🔁</td>
    <td>True / False</td>
    <td>Claim verification</td>
    <td>Open (CC BY 4.0)</td>
  </tr>

  <tr>
    <td>ChronoClaims <a href="#ref86">[86]</a></td>
    <td>English</td>
    <td>T</td>
    <td>✅ event timelines</td>
    <td>Support / Refute</td>
    <td>Temporal verification</td>
    <td>Not public</td>
  </tr>

  <tr>
    <td>QuanTemp <a href="#ref102">[102]</a></td>
    <td>English</td>
    <td>T, E</td>
    <td>✅ temporal claims</td>
    <td>Fact-checker ratings</td>
    <td>Numerical claims</td>
    <td>Open (CC BY-NC)</td>
  </tr>

  <tr>
    <td>TSVer <a href="#ref103">[103]</a></td>
    <td>English</td>
    <td>T + time series</td>
    <td>✅</td>
    <td>Verdict + justification</td>
    <td>Time-series verification</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>M4FC <a href="#ref104">[104]</a></td>
    <td>10 languages</td>
    <td>T, I</td>
    <td>✅ temporal split</td>
    <td>6 tasks incl. location, verdict</td>
    <td>Multitask MM fact-checking</td>
    <td>Open (CC BY-SA)</td>
  </tr>

  <tr>
    <td>MMM-Fact <a href="#ref105">[105]</a></td>
    <td>English</td>
    <td>T, I, V, tables, E</td>
    <td>✅ 1995–2025</td>
    <td>True / False / NEI</td>
    <td>Veracity + explanation</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>AVerImaTeC <a href="#ref106">[106]</a></td>
    <td>English</td>
    <td>T, I, E (QA)</td>
    <td>✅ pre-claim evidence</td>
    <td>S / R / NEE / CE-CP</td>
    <td>Image–text verification</td>
    <td>Open (CC BY-NC)</td>
  </tr>

  <tr>
    <td>5Pils <a href="#ref107">[107]</a></td>
    <td>English</td>
    <td>I, T</td>
    <td>✅ date pillar</td>
    <td>Source, date, location, motivation, type</td>
    <td>Image meta-context</td>
    <td>Open (CC BY-SA)</td>
  </tr>

  <tr>
    <td>MOCHEG <a href="#ref109">[109]</a></td>
    <td>English</td>
    <td>T, I, E</td>
    <td>✅ via fact-check date</td>
    <td>S / R / NEI + explanation</td>
    <td>End-to-end MM fact-checking</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>VERITE <a href="#ref110">[110]</a></td>
    <td>English</td>
    <td>T, I</td>
    <td>-</td>
    <td>True / OOC / Miscaptioned</td>
    <td>OOC detection</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>X-POSE <a href="#ref96">[96]</a></td>
    <td>English</td>
    <td>T, I, E</td>
    <td>✅ 2017–2025</td>
    <td>Truthful / Misinformation</td>
    <td>Image–text verification</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>MultiClaim <a href="#ref111">[111]</a></td>
    <td>27 + 39 languages</td>
    <td>T, I (OCR)</td>
    <td>✅ post dates</td>
    <td>Fact-checker ratings</td>
    <td>Claim retrieval</td>
    <td>On request</td>
  </tr>

  <tr>
    <td>MR2 <a href="#ref113">[113]</a></td>
    <td>English, Chinese</td>
    <td>T, I, S, E</td>
    <td>~ (post metadata)</td>
    <td>Rumor / Non-Rumor / Unverified</td>
    <td>Retrieval-augmented rumour detection</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>MCFEND <a href="#ref126">[126]</a></td>
    <td>Chinese</td>
    <td>T, I, S, Src</td>
    <td>✅ post times</td>
    <td>Real / Fake</td>
    <td>Multi-source fake news</td>
    <td>See repo</td>
  </tr>

  <tr>
    <td>NewsCLIPpings <a href="#ref114">[114]</a></td>
    <td>English</td>
    <td>T, I</td>
    <td>~ (VisualNews dates)</td>
    <td>Pristine / Falsified</td>
    <td>OOC detection</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>DGM4 <a href="#ref80">[80]</a></td>
    <td>English</td>
    <td>T, I</td>
    <td>-</td>
    <td>Manipulation type + grounding</td>
    <td>Manipulation detection/grounding</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>MMFakeBench <a href="#ref85">[85]</a></td>
    <td>English</td>
    <td>T, I</td>
    <td>-</td>
    <td>3 sources, 12 subtypes</td>
    <td>LVLM misinformation detection</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>FakeNewsNet <a href="#ref118">[118]</a></td>
    <td>English</td>
    <td>T, I, S, Src</td>
    <td>✅ engagement timestamps</td>
    <td>Fake / Real</td>
    <td>News classification + propagation</td>
    <td>Rehydration</td>
  </tr>

  <tr>
    <td>UPFD <a href="#ref77">[77]</a></td>
    <td>English</td>
    <td>T, S (graph)</td>
    <td>~ (propagation order)</td>
    <td>Fake / Real</td>
    <td>Graph classification</td>
    <td>Open (PyG)</td>
  </tr>

  <tr>
    <td>MuMiN <a href="#ref119">[119]</a></td>
    <td>41 languages</td>
    <td>T, I, S, Src</td>
    <td>✅ ~10 years</td>
    <td>Misinformation / Factual</td>
    <td>Heterogeneous graph classification</td>
    <td>Rehydration</td>
  </tr>

  <tr>
    <td>NELA-GT-2022 <a href="#ref120">[120]</a></td>
    <td>English</td>
    <td>T, Src</td>
    <td>✅ daily, full year</td>
    <td>Source reliability</td>
    <td>Source-level credibility</td>
    <td>Open (Dataverse)</td>
  </tr>

  <tr>
    <td>FineFake <a href="#ref121">[121]</a></td>
    <td>English</td>
    <td>T, I, KG</td>
    <td>✅ publication date</td>
    <td>Binary + 6 fine-grained</td>
    <td>Knowledge-aware fake news</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>Twitter15/16 <a href="#ref127">[127]</a></td>
    <td>English</td>
    <td>T, S (trees)</td>
    <td>✅ time delays</td>
    <td>NR / FR / TR / UR</td>
    <td>Rumour propagation</td>
    <td>Mirrors</td>
  </tr>

  <tr>
    <td>PHEME <a href="#ref128">[128]</a></td>
    <td>English</td>
    <td>T, S (threads)</td>
    <td>✅</td>
    <td>Rumour / Non-rumour (+veracity)</td>
    <td>Rumour detection</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>Fakeddit <a href="#ref130">[130]</a></td>
    <td>English</td>
    <td>T, I, S (comments)</td>
    <td>✅ 2008–2019</td>
    <td>2 / 3 / 6-way</td>
    <td>Fine-grained fake news</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>FakeSV <a href="#ref81">[81]</a></td>
    <td>Chinese</td>
    <td>V, A, T, S</td>
    <td>✅ temporal split</td>
    <td>Fake / Real (+debunk)</td>
    <td>Short-video fake news</td>
    <td>DUA</td>
  </tr>

  <tr>
    <td>FakeTT <a href="#ref83">[83]</a></td>
    <td>English</td>
    <td>V, A, T</td>
    <td>✅ 2018–2024</td>
    <td>Fake / Real</td>
    <td>Short-video fake news</td>
    <td>Application</td>
  </tr>

  <tr>
    <td>FMNV <a href="#ref133">[133]</a></td>
    <td>English</td>
    <td>V, A, T</td>
    <td>✅ 5 years</td>
    <td>Real + 4 fake types</td>
    <td>News-video fake news</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>RAVM <a href="#ref92">[92]</a></td>
    <td>-</td>
    <td>V, A, T</td>
    <td>-</td>
    <td>Real / Fake + intent polarity + attribution</td>
    <td>Generated-video misinformation</td>
    <td>Unavailable (link offline)</td>
  </tr>

  <tr>
    <td>FVC <a href="#ref134">[134]</a></td>
    <td>EN, FR, RU, DE, AR</td>
    <td>V, S (near-dups)</td>
    <td>✅ re-upload cascades</td>
    <td>Fake / Real</td>
    <td>Video verification</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>AVeriTeC <a href="#ref137">[137]</a></td>
    <td>English</td>
    <td>T, E (QA)</td>
    <td>✅ pre-claim evidence</td>
    <td>S / R / NEE / CE-CP</td>
    <td>Real-world claim verification</td>
    <td>Open</td>
  </tr>

  <tr>
    <td>LIAR <a href="#ref138">[138]</a></td>
    <td>English</td>
    <td>T, Src (speaker)</td>
    <td>~ (statement date)</td>
    <td>6-way truthfulness</td>
    <td>Statement classification</td>
    <td>Open</td>
  </tr>

  </tbody>
</table>
</div>

---

## 📊 Graph-Learning Benchmarks (Temporal / Dynamic)

Use these to test the dynamic-graph backbone of a graph world model before applying it to fact verification.

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>Year</th>
      <th>Venue</th>
      <th>What it measures</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>

  <tr>
    <td>TGB <a href="#ref145">[145]</a></td>
    <td>2023</td>
    <td>NeurIPS D&amp;B</td>
    <td>Dynamic link / node property prediction on large real temporal graphs (e.g. tgbl-wiki, tgbl-review, tgbl-coin, tgbl-flight)</td>
    <td><a href="https://tgb.complexdatalab.com/">Website</a></td>
  </tr>

  <tr>
    <td>TGB 2.0 <a href="#ref146">[146]</a></td>
    <td>2024</td>
    <td>NeurIPS D&amp;B</td>
    <td>Adds temporal knowledge graphs (tkgl-*) and temporal heterogeneous graphs (thgl-*)</td>
    <td><a href="https://tgb.complexdatalab.com/">Website</a></td>
  </tr>

  <tr>
    <td>TGB-Seq <a href="#ref55">[55]</a></td>
    <td>2025</td>
    <td>ICLR</td>
    <td>Low edge-repetition datasets that defeat memorisation; tests sequential dynamics</td>
    <td><a href="https://github.com/TGB-Seq/TGB-Seq">Official</a></td>
  </tr>

  <tr>
    <td>DTGB <a href="#ref147">[147]</a></td>
    <td>2024</td>
    <td>NeurIPS D&amp;B</td>
    <td>Dynamic <em>text-attributed</em> graphs: node and edge text that changes over time</td>
    <td><a href="https://github.com/zjs123/DTGB">Official</a></td>
  </tr>

  <tr>
    <td>GDGB <a href="#ref148">[148]</a></td>
    <td>2026</td>
    <td>ICLR</td>
    <td><em>Generative</em> dynamic text-attributed graph learning: the closest existing benchmark for predicting future graphs</td>
    <td>-</td>
  </tr>

  <tr>
    <td>EdgeBank / DGB <a href="#ref46">[46]</a></td>
    <td>2022</td>
    <td>NeurIPS D&amp;B</td>
    <td>Harder negative sampling (historical / inductive) for dynamic link prediction</td>
    <td>-</td>
  </tr>

  <tr>
    <td>ICEWS / GDELT / WIKI / YAGO <a href="#ref44">[44]</a></td>
    <td>—</td>
    <td>—</td>
    <td>Standard TKG forecasting datasets (event graphs with daily or yearly timestamps); used by RE-NET, RE-GCN, TLogic, etc.</td>
    <td>-</td>
  </tr>

  </tbody>
</table>
</div>

---

## 🛠️ Tools, Libraries & APIs

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6">
  <thead><tr><th>Layer</th><th>Tool</th><th>Notes</th></tr></thead>
  <tbody>
  <tr><td>Graph learning</td><td><a href="https://github.com/pyg-team/pytorch_geometric">PyTorch Geometric</a> · <a href="https://github.com/dmlc/dgl">DGL</a></td><td>Use <code>HeteroData</code> for typed evidence graphs; PyG ships <code>UPFD</code> <a href="#ref77">[77]</a>.</td></tr>
  <tr><td>Dynamic graphs</td><td><a href="https://github.com/yule-BUAA/DyGLib">DyGLib</a> · <a href="https://github.com/shenyangHuang/TGB">TGB</a> · <a href="https://github.com/benedekrozemberczki/pytorch_geometric_temporal">PyG Temporal</a> <a href="#ref149">[149]</a></td><td>Most of these assume ID-only or low-dimensional features; rich multimodal node attributes usually need custom code.</td></tr>
  <tr><td>Temporal KG memory</td><td><a href="https://github.com/getzep/graphiti">Graphiti</a> <a href="#ref16">[16]</a></td><td>Bi-temporal edges (valid time vs. ingestion time); a useful reference design for evidence graphs that must record both "when it was true" and "when we learned it".</td></tr>
  <tr><td>Graph world models</td><td><a href="https://github.com/ulab-uiuc/GWM">GWM</a> · <a href="https://github.com/tkipf/c-swm">C-SWM</a> · <a href="https://github.com/USTC-DataDarknessLab/Graph-Native_World_Modeling">WorldGraph</a> · <a href="https://github.com/Scarlett-Yyq/World-as-Graph">WaG</a></td><td>Reference implementations for action-conditioned graph transitions.</td></tr>
  <tr><td>Agentic fact-checking</td><td><a href="https://github.com/multimodal-ai-lab/DEFAME">DEFAME</a> <a href="#ref87">[87]</a></td><td>Strong multimodal baseline (web, image and reverse-image search, geolocation).</td></tr>
  <tr><td>Uncertainty</td><td><a href="https://github.com/zxj32/uncertainty-GNN">uncertainty-GNN</a> · <a href="https://github.com/snap-stanford/conformalized-gnn">CF-GNN</a> · <a href="https://github.com/ODYSSEYWT/NCPNET">NCPNet</a></td><td>Vacuity / dissonance heads and conformal prediction sets.</td></tr>
  <tr><td>Fact-check feeds</td><td><a href="https://developers.google.com/fact-check/tools/api">Google Fact Check Tools API</a> (ClaimReview)</td><td>Dated verdicts across many publishers; free with an API key.</td></tr>
  <tr><td>Temporal evidence</td><td><a href="https://archive.org/help/wayback_api.php">Wayback Machine / CDX API</a> · <a href="https://www.gdeltproject.org/data.html">GDELT</a> · <a href="https://zenodo.org/records/5792475">Wikidated 1.0</a></td><td>Reconstruct what a page said at time <em>t</em>; global event streams; Wikidata history as a sequence of KG deltas.</td></tr>
  <tr><td>Web / reverse search</td><td><a href="https://brave.com/search/api/">Brave Search API</a> · <a href="https://tavily.com/">Tavily</a> · <a href="https://serpapi.com/google-lens-api">SerpApi Google Lens</a></td><td>The Bing Search APIs were <a href="https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement">retired on 11 Aug 2025</a>, so older pipelines that rely on them will not reproduce.</td></tr>
  <tr><td>Near-duplicate media</td><td><a href="https://github.com/facebookresearch/faiss">FAISS</a> · <a href="https://github.com/FeipengMa6/VSC22-Submission">VSC22 winning solution</a></td><td>Build DERIVED-FROM / NEAR-DUPLICATE edges between images and videos.</td></tr>
  <tr><td>Provenance & forensics</td><td><a href="https://opensource.contentauthenticity.org/docs/c2pa-python/">c2pa-python</a> · <a href="https://github.com/SCLBD/DeepfakeBench">DeepfakeBench</a></td><td>Signed C2PA manifests are high-confidence provenance edges.</td></tr>
  <tr><td>Open multimodal LLMs</td><td><a href="https://github.com/QwenLM/Qwen3-VL">Qwen3-VL</a></td><td>Apache-2.0; long context, video timestamp grounding; suitable for extracting graph elements.</td></tr>
  </tbody>
</table>
</div>

---

## 📚 Surveys & Further Reading

- **Graph world models.** Liu et al., *Graph World Models: Concepts, Taxonomy, and Future Directions* <a href="#ref1">[1]</a>. Song & Cai, *Understanding Rollout Error in Graph World Models* <a href="#ref23">[23]</a>.
- **World models (general).** Ding et al., ACM CSUR survey <a href="#ref33">[33]</a>; *Critiques of World Models* <a href="#ref34">[34]</a>.
- **Graph memory for agents.** Yang et al. <a href="#ref35">[35]</a>.
- **Dynamic GNNs.** Zheng, Yi & Wei <a href="#ref150">[150]</a>.
- **Uncertainty in GNNs.** Wang et al. <a href="#ref63">[63]</a>.
- **Automated fact-checking.** Guo, Schlichtkrull & Vlachos <a href="#ref151">[151]</a>; multimodal AFC survey by Akhtar et al. <a href="#ref152">[152]</a>.
- **Graph-based fake-news detection.** <a href="#ref153">[153]</a>.
- **Misinformation videos.** Qi et al. <a href="#ref154">[154]</a> (with the companion [Awesome-Misinfo-Video-Detection](https://github.com/ICTMCG/Awesome-Misinfo-Video-Detection) list).
- **Misinformation datasets.** Thibault et al. <a href="#ref144">[144]</a>.

---

## 📖 References

<a id="ref1"></a>[1] J. Liu, S. Yang, M. Wang, Y. Wang, B. Yu, "Graph World Models: Concepts, Taxonomy, and Future Directions," *arXiv:2604.27895*, 2026. [Paper](https://arxiv.org/abs/2604.27895)

<a id="ref2"></a>[2] D. Ha, J. Schmidhuber, "World Models," *arXiv:1803.10122*, 2018. [Paper](https://arxiv.org/abs/1803.10122) | [Project](https://worldmodels.github.io/)

<a id="ref3"></a>[3] P. Battaglia et al., "Interaction Networks for Learning about Objects, Relations and Physics," *NeurIPS*, 2016. [Paper](https://arxiv.org/abs/1612.00222)

<a id="ref4"></a>[4] A. Sanchez-Gonzalez et al., "Graph Networks as Learnable Physics Engines for Inference and Control," *ICML*, 2018. [Paper](https://arxiv.org/abs/1806.01242)

<a id="ref5"></a>[5] P. Battaglia et al., "Relational Inductive Biases, Deep Learning, and Graph Networks," *arXiv:1806.01261*, 2018. [Paper](https://arxiv.org/abs/1806.01261) | [Code](https://github.com/google-deepmind/graph_nets)

<a id="ref6"></a>[6] N. Savinov, A. Dosovitskiy, V. Koltun, "Semi-Parametric Topological Memory for Navigation," *ICLR*, 2018. [Paper](https://arxiv.org/abs/1803.00653)

<a id="ref7"></a>[7] B. Eysenbach, R. Salakhutdinov, S. Levine, "Search on the Replay Buffer: Bridging Planning and Reinforcement Learning," *NeurIPS*, 2019. [Paper](https://arxiv.org/abs/1906.05253)

<a id="ref8"></a>[8] T. Kipf, E. van der Pol, M. Welling, "Contrastive Learning of Structured World Models," *ICLR*, 2020. [Paper](https://arxiv.org/abs/1911.12247) | [Code](https://github.com/tkipf/c-swm)

<a id="ref9"></a>[9] Z. Lin et al., "Improving Generative Imagination in Object-Centric World Models (G-SWM)," *ICML*, 2020. [Paper](https://arxiv.org/abs/2010.02054)

<a id="ref10"></a>[10] A. Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph Networks (GNS)," *ICML*, 2020. [Paper](https://arxiv.org/abs/2002.09405) | [Code](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate)

<a id="ref11"></a>[11] T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P. Battaglia, "Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets)," *ICLR*, 2021. [Paper](https://arxiv.org/abs/2010.03409) | [Code](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)

<a id="ref12"></a>[12] L. Zhang, G. Yang, B. Stadie, "World Model as a Graph: Learning Latent Landmarks for Planning (L3P)," *ICML*, 2021. [Paper](https://arxiv.org/abs/2011.12491)

<a id="ref13"></a>[13] P. Ammanabrolu, M. Riedl, "Learning Knowledge Graph-based World Models of Textual Environments (Worldformer)," *NeurIPS*, 2021. [Paper](https://proceedings.neurips.cc/paper/2021/file/1e747ddbea997a1b933aaf58a7953c3c-Paper.pdf)

<a id="ref14"></a>[14] R. Lam et al., "Learning Skillful Medium-Range Global Weather Forecasting (GraphCast)," *Science, 382(6677)*, 2023. [Paper](https://www.science.org/doi/10.1126/science.adi2336) | [Code](https://github.com/google-deepmind/graphcast)

<a id="ref15"></a>[15] P. Anokhin et al., "AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents," *IJCAI*, 2025. [Paper](https://arxiv.org/abs/2407.04363) | [Code](https://github.com/AIRI-Institute/AriGraph)

<a id="ref16"></a>[16] P. Rasmussen et al., "Zep: A Temporal Knowledge Graph Architecture for Agent Memory," *arXiv:2501.13956*, 2025. [Paper](https://arxiv.org/abs/2501.13956) | [Code](https://github.com/getzep/graphiti)

<a id="ref17"></a>[17] T. Feng, Y. Wu, G. Lin, J. You, "Graph World Model," *ICML (PMLR v267)*, 2025. [Paper](https://arxiv.org/abs/2507.10539) | [Code](https://github.com/ulab-uiuc/GWM)

<a id="ref18"></a>[18] Z. Wang, K. Wang, L. Zhao, P. Stone, J. Bian, "Dyn-O: Building Structured World Models with Object-Centric Representations," *NeurIPS*, 2025. [Paper](https://arxiv.org/abs/2507.03298)

<a id="ref19"></a>[19] F. Feng, P. Lippe, S. Magliacane, "Learning Interactive World Model for Object-Centric Reinforcement Learning (FIOC-WM)," *NeurIPS*, 2025. [Paper](https://arxiv.org/abs/2511.02225)

<a id="ref20"></a>[20] Hu et al., "Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation," *arXiv:2508.06990*, 2025. [Paper](https://arxiv.org/abs/2508.06990)

<a id="ref21"></a>[21] H. Liu, Y. Wei, F. Xing, T. Derr, H. Han, Y. Zhang, "Graph2Video: Leveraging Video Models to Model Dynamic Graph Evolution," *AAAI*, 2026. [Paper](https://arxiv.org/abs/2603.13360) | [Code](https://github.com/hualiu829/Graph2Video)

<a id="ref22"></a>[22] H. Nam, Q. Le Lidec, L. Maes, Y. LeCun, R. Balestriero, "Causal-JEPA: Learning World Models through Object-Level Latent Masking," *ICML (PMLR 306)*, 2026. [Paper](https://arxiv.org/abs/2602.11389) | [Code](https://github.com/galilai-group/cjepa)

<a id="ref23"></a>[23] X. Song, Z. Cai, "Understanding Rollout Error in Graph World Models," *arXiv:2606.27780*, 2026. [Paper](https://arxiv.org/abs/2606.27780) | [Code](https://github.com/Hik289/graph_world_model_accumulative_error)

<a id="ref24"></a>[24] W. Wang, Y. Chen, H. Yang, Y. Liu, M. Luo, X. Jiao, X. Wen, M. Liu, "A Structural Dynamics Graph World Model: Unified Modeling, Constrained Rollout, and Interpretable Calibration (SD-GWM)," *arXiv:2608.08689*, 2026. [Paper](https://arxiv.org/abs/2608.08689)

<a id="ref25"></a>[25] X. Song, Z. Cai, "Repair the Amplifier, Not the Symptom: Stable World-Model Correction for Agent Rollouts," *arXiv:2607.01767*, 2026. [Paper](https://arxiv.org/abs/2607.01767) | [Code](https://github.com/Hik289/world-model-corrector)

<a id="ref26"></a>[26] Z. Ding, Y. Li, X. Xie, "WorldGraph: Graph-Native World Modeling," *arXiv:2609.34159*, 2026. [Paper](https://arxiv.org/abs/2609.34159) | [Code](https://github.com/USTC-DataDarknessLab/Graph-Native_World_Modeling)

<a id="ref27"></a>[27] Y. Yang, S. Huang, Y. Huang, F. Ke, J. Han, X. Zheng, "World-as-Graph: Relational World Modeling Through Latent Space Graphs (WaG)," *arXiv:2609.38927*, 2026. [Paper](https://arxiv.org/abs/2609.38927) | [Code](https://github.com/Scarlett-Yyq/World-as-Graph)

<a id="ref28"></a>[28] D. Hafner et al., "Mastering Diverse Control Tasks through World Models (DreamerV3)," *Nature*, 2025. [Paper](https://arxiv.org/abs/2301.04104) | [Code](https://github.com/danijar/dreamerv3)

<a id="ref29"></a>[29] M. Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," *arXiv:2506.09985*, 2025. [Paper](https://arxiv.org/abs/2506.09985)

<a id="ref30"></a>[30] G. Zhou, H. Pan, Y. LeCun, L. Pinto, "DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning," *ICML*, 2025. [Paper](https://arxiv.org/abs/2411.04983)

<a id="ref31"></a>[31] Y. Gu et al., "Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents (WebDreamer)," *arXiv:2411.06559*, 2024. [Paper](https://arxiv.org/abs/2411.06559) | [Code](https://github.com/OSU-NLP-Group/WebDreamer)

<a id="ref32"></a>[32] K. Mei et al., "R-WoM: Retrieval-augmented World Model for Computer-use Agents," *ICLR*, 2026. [Paper](https://arxiv.org/abs/2510.11892)

<a id="ref33"></a>[33] J. Ding et al., "Understanding World or Predicting Future? A Comprehensive Survey of World Models," *ACM Computing Surveys 58(3)*, 2025. [Paper](https://arxiv.org/abs/2411.14499)

<a id="ref34"></a>[34] E. Xing et al., "Critiques of World Models," *arXiv:2507.05169*, 2025. [Paper](https://arxiv.org/abs/2507.05169)

<a id="ref35"></a>[35] C. Yang et al., "Graph-based Agent Memory: Taxonomy, Techniques, and Applications," *arXiv:2602.05665*, 2026. [Paper](https://arxiv.org/abs/2602.05665) | [Code](https://github.com/DEEP-PolyU/Awesome-GraphMemory)

<a id="ref36"></a>[36] S. Kumar, X. Zhang, J. Leskovec, "Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks (JODIE)," *KDD*, 2019. [Paper](https://arxiv.org/abs/1908.01207) | [Code](https://github.com/claws-lab/jodie)

<a id="ref37"></a>[37] E. Hajiramezanali et al., "Variational Graph Recurrent Neural Networks (VGRNN)," *NeurIPS*, 2019. [Paper](https://arxiv.org/abs/1908.09710) | [Code](https://github.com/VGraphRNN/VGRNN)

<a id="ref38"></a>[38] D. Xu et al., "Inductive Representation Learning on Temporal Graphs (TGAT)," *ICLR*, 2020. [Paper](https://arxiv.org/abs/2002.07962)

<a id="ref39"></a>[39] E. Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs (TGN)," *arXiv:2006.10637*, 2020. [Paper](https://arxiv.org/abs/2006.10637) | [Code](https://github.com/twitter-research/tgn)

<a id="ref40"></a>[40] Y. Wang et al., "Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks (CAWN)," *ICLR*, 2021. [Paper](https://arxiv.org/abs/2101.05974)

<a id="ref41"></a>[41] W. Jin, M. Qu, X. Jin, X. Ren, "Recurrent Event Network: Autoregressive Structure Inference over Temporal Knowledge Graphs (RE-NET)," *EMNLP*, 2020. [Paper](https://arxiv.org/abs/1904.05530) | [Code](https://github.com/INK-USC/RE-Net)

<a id="ref42"></a>[42] C. Zhu et al., "Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks (CyGNet)," *AAAI*, 2021. [Paper](https://arxiv.org/abs/2012.08492)

<a id="ref43"></a>[43] Z. Han et al., "Explainable Subgraph Reasoning for Forecasting on Temporal Knowledge Graphs (xERTE)," *ICLR*, 2021. [Paper](https://arxiv.org/abs/2012.15537)

<a id="ref44"></a>[44] Z. Li et al., "Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning (RE-GCN)," *SIGIR*, 2021. [Paper](https://arxiv.org/abs/2104.10353) | [Code](https://github.com/Lee-zix/RE-GCN)

<a id="ref45"></a>[45] Y. Liu et al., "TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs," *AAAI*, 2022. [Paper](https://arxiv.org/abs/2112.08025) | [Code](https://github.com/liu-yushan/TLogic)

<a id="ref46"></a>[46] F. Poursafaei et al., "Towards Better Evaluation for Dynamic Link Prediction (EdgeBank)," *NeurIPS Datasets & Benchmarks*, 2022. [Paper](https://arxiv.org/abs/2207.10128)

<a id="ref47"></a>[47] W. Cong et al., "Do We Really Need Complicated Model Architectures for Temporal Networks? (GraphMixer)," *ICLR*, 2023. [Paper](https://arxiv.org/abs/2302.11636)

<a id="ref48"></a>[48] L. Yu, L. Sun, B. Du, W. Lv, "Towards Better Dynamic Graph Learning: New Architecture and Unified Library (DyGFormer / DyGLib)," *NeurIPS*, 2023. [Paper](https://arxiv.org/abs/2303.13047) | [Code](https://github.com/yule-BUAA/DyGLib)

<a id="ref49"></a>[49] R. Liao et al., "GenTKG: Generative Forecasting on Temporal Knowledge Graph with Large Language Models," *Findings of NAACL*, 2024. [Paper](https://arxiv.org/abs/2310.07793)

<a id="ref50"></a>[50] J. Gastinger et al., "History Repeats Itself: A Baseline for Temporal Knowledge Graph Forecasting," *IJCAI*, 2024. [Paper](https://arxiv.org/abs/2404.16726) | [Code](https://github.com/nec-research/recurrency_baseline_tkg)

<a id="ref51"></a>[51] F. Cornell et al., "On the Power of Heuristics in Temporal Graphs," *arXiv:2502.04910*, 2025. [Paper](https://arxiv.org/abs/2502.04910)

<a id="ref52"></a>[52] J. Gastinger, C. Meilicke, H. Stuckenschmidt, "CountTRuCoLa: Rule Confidence Learning for Temporal Knowledge Graph Forecasting," *arXiv:2509.09474*, 2025. [Paper](https://arxiv.org/abs/2509.09474) | [Code](https://github.com/JuliaGast/counttrucola)

<a id="ref53"></a>[53] A. Hayes, T. Schumacher, M. Strohmaier, "What Do Temporal Graph Learning Models Learn?," *arXiv:2510.09416*, 2025. [Paper](https://arxiv.org/abs/2510.09416)

<a id="ref54"></a>[54] C. Hu et al., "CFEP: Conformal Event Prediction with Temporal Knowledge Graph," *Findings of ACL*, 2026. [Paper](https://aclanthology.org/2026.findings-acl.258.pdf) | [Code](https://github.com/hucheng-IIE/CFEP)

<a id="ref55"></a>[55] L. Yi et al., "TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics," *ICLR*, 2025. [Paper](https://arxiv.org/abs/2502.02975) | [Code](https://github.com/TGB-Seq/TGB-Seq)

<a id="ref56"></a>[56] X. L. Dong, L. Berti-Équille, D. Srivastava, "Integrating Conflicting Data: The Role of Source Dependence," *VLDB*, 2009. [Paper](https://lunadong.com/publication/dependence_vldb.pdf)

<a id="ref57"></a>[57] X. Zhao, F. Chen, S. Hu, J.-H. Cho, "Uncertainty Aware Semi-Supervised Learning on Graph Data (S-BGCN / GKDE)," *NeurIPS*, 2020. [Paper](https://arxiv.org/abs/2010.12783) | [Code](https://github.com/zxj32/uncertainty-GNN)

<a id="ref58"></a>[58] M. Stadler et al., "Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification," *NeurIPS*, 2021. [Paper](https://arxiv.org/abs/2110.14012) | [Code](https://github.com/stadlmax/Graph-Posterior-Network)

<a id="ref59"></a>[59] X. Wang et al., "Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration (CaGCN)," *NeurIPS*, 2021. [Paper](https://arxiv.org/abs/2109.14285)

<a id="ref60"></a>[60] K. Huang, Y. Jin, E. Candès, J. Leskovec, "Uncertainty Quantification over Graph with Conformalized Graph Neural Networks (CF-GNN)," *NeurIPS*, 2023. [Paper](https://arxiv.org/abs/2305.14535) | [Code](https://github.com/snap-stanford/conformalized-gnn)

<a id="ref61"></a>[61] F. Bickford Smith et al., "Prediction-Oriented Bayesian Active Learning (EPIG)," *AISTATS*, 2023. [Paper](https://arxiv.org/abs/2304.08151)

<a id="ref62"></a>[62] V. Bengs et al., "Is Epistemic Uncertainty Faithfully Represented by Evidential Deep Learning Methods?," *ICML*, 2024. [Paper](https://arxiv.org/abs/2402.09056)

<a id="ref63"></a>[63] F. Wang et al., "Uncertainty in Graph Neural Networks: A Survey," *TMLR*, 2024. [Paper](https://arxiv.org/abs/2403.07185)

<a id="ref64"></a>[64] T. Wang et al., "Non-exchangeable Conformal Prediction for Temporal Graph Neural Networks (NCPNet)," *KDD*, 2025. [Paper](https://arxiv.org/abs/2507.02151) | [Code](https://github.com/ODYSSEYWT/NCPNET)

<a id="ref65"></a>[65] "BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design," *arXiv:2508.21184*, 2025. [Paper](https://arxiv.org/abs/2508.21184)

<a id="ref66"></a>[66] Lan et al., "Multi-Evidence based Fact Verification via a Confidential Graph Neural Network (CO-GAT)," *arXiv:2405.10481*, 2024. [Paper](https://arxiv.org/abs/2405.10481) | [Code](https://github.com/NEUIR/CO-GAT)

<a id="ref67"></a>[67] "INFOGATHERER: Principled Information Seeking via Evidence Retrieval and Strategic Questioning," *arXiv:2603.05909*, 2026. [Paper](https://arxiv.org/abs/2603.05909)

<a id="ref68"></a>[68] J. Zhou et al., "GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification," *ACL*, 2019. [Paper](https://arxiv.org/abs/1908.01843) | [Code](https://github.com/thunlp/GEAR)

<a id="ref69"></a>[69] Z. Liu et al., "Fine-grained Fact Verification with Kernel Graph Attention Network (KGAT)," *ACL*, 2020. [Paper](https://arxiv.org/abs/1910.09796) | [Code](https://github.com/thunlp/KernelGAT)

<a id="ref70"></a>[70] W. Zhong et al., "Reasoning Over Semantic-Level Graph for Fact Checking (DREAM)," *ACL*, 2020. [Paper](https://arxiv.org/abs/1909.03745)

<a id="ref71"></a>[71] J. Kim et al., "FactKG: Fact Verification via Reasoning on Knowledge Graphs," *ACL*, 2023. [Paper](https://arxiv.org/abs/2305.06590)

<a id="ref72"></a>[72] J. Ma, W. Gao, K.-F. Wong, "Rumor Detection on Twitter with Tree-structured Recursive Neural Networks," *ACL*, 2018. [Paper](https://aclanthology.org/P18-1184.pdf)

<a id="ref73"></a>[73] T. Bian et al., "Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks (BiGCN)," *AAAI*, 2020. [Paper](https://arxiv.org/abs/2001.06362) | [Code](https://github.com/TianBian95/BiGCN)

<a id="ref74"></a>[74] Y.-J. Lu, C.-T. Li, "GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media," *ACL*, 2020. [Paper](https://arxiv.org/abs/2004.11648)

<a id="ref75"></a>[75] V.-H. Nguyen, K. Sugiyama, P. Nakov, M.-Y. Kan, "FANG: Leveraging Social Context for Fake News Detection Using Graph Representation," *CIKM*, 2020. [Paper](https://arxiv.org/abs/2008.07939)

<a id="ref76"></a>[76] L. Wei et al., "Towards Propagation Uncertainty: Edge-enhanced Bayesian Graph Convolutional Networks for Rumor Detection (EBGCN)," *ACL*, 2021. [Paper](https://arxiv.org/abs/2107.11934)

<a id="ref77"></a>[77] Y. Dou et al., "User Preference-aware Fake News Detection (UPFD)," *SIGIR*, 2021. [Paper](https://arxiv.org/abs/2104.12259) | [Code](https://github.com/safe-graph/GNN-FakeNews) | [PyG dataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.UPFD.html)

<a id="ref78"></a>[78] Z. Yang et al., "A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection (CofCED; LIAR-RAW & RAWFC)," *COLING*, 2022. [Paper](https://aclanthology.org/2022.coling-1.230/) | [Code](https://github.com/Nicozwy/CofCED)

<a id="ref79"></a>[79] L. Allein, I. Augenstein, M.-F. Moens, "Time-Aware Evidence Ranking for Fact-Checking," *arXiv:2009.06402*, 2020. [Paper](https://arxiv.org/abs/2009.06402)

<a id="ref80"></a>[80] R. Shao, T. Wu, Z. Liu, "Detecting and Grounding Multi-Modal Media Manipulation (DGM4 / HAMMER)," *CVPR*, 2023. [Paper](https://arxiv.org/abs/2304.02556) | [Code](https://github.com/rshaojimmy/MultiModal-DeepFake)

<a id="ref81"></a>[81] P. Qi et al., "FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms (SV-FEND)," *AAAI*, 2023. [Paper](https://arxiv.org/abs/2211.10973) | [Code](https://github.com/ICTMCG/FakeSV)

<a id="ref82"></a>[82] P. Qi et al., "SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection," *CVPR*, 2024. [Paper](https://arxiv.org/abs/2403.03170)

<a id="ref83"></a>[83] Y. Bu et al., "FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process (FakeTT)," *ACM MM*, 2024. [Paper](https://arxiv.org/abs/2407.16670) | [Code](https://github.com/ICTMCG/FakingRecipe)

<a id="ref84"></a>[84] J. Kim, J. Lee, Y. In, K. Yoon, C. Park, "Revisiting Fake News Detection: Towards Temporality-aware Evaluation by Leveraging Engagement Earliness (DAWN)," *WSDM*, 2025. [Paper](https://doi.org/10.1145/3701551.3703524) | [Code](https://github.com/LeeJunmo/DAWN) | [arXiv](https://arxiv.org/abs/2411.12775) | [Follow-up ACM article: A New DAWN for Fake News Detection](https://dl.acm.org/doi/10.1145/3815193)

<a id="ref85"></a>[85] X. Liu et al., "MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs," *ICLR*, 2025. [Paper](https://arxiv.org/abs/2406.08772) | [Code](https://github.com/liuxuannan/MMFakeBench)

<a id="ref86"></a>[86] A. M. Barik, W. Hsu, M. L. Lee, "ChronoFact: Timeline-based Temporal Fact Verification (ChronoClaims)," *IJCAI*, 2025. [Paper](https://www.ijcai.org/proceedings/2025/0893.pdf) | [arXiv](https://arxiv.org/abs/2410.14964)

<a id="ref87"></a>[87] T. Braun, M. Rothermel, M. Rohrbach, A. Rohrbach, "DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts," *ICML*, 2025. [Paper](https://arxiv.org/abs/2412.10510) | [Code](https://github.com/multimodal-ai-lab/DEFAME)

<a id="ref88"></a>[88] M. Fu et al., "Seeking and Updating with Live Visual Knowledge (LiveVQA)," *NeurIPS*, 2025. [Paper](https://arxiv.org/abs/2504.05288) | [Code](https://github.com/fumingyang2004/LIVEVQA)

<a id="ref89"></a>[89] J. Hu, J. Zhang, Z. Li, "Tracing Truth: Dynamic Temporal Networks for Multi-modal Fake News Detection (DTN)," *PeerJ Computer Science*, 2025. [Paper](https://doi.org/10.7717/peerj-cs.2998)

<a id="ref90"></a>[90] "Evidence-Grounded Multimodal Misinformation Detection with Attention-Based GNNs (EGMMG)," *arXiv:2505.18221*, 2025. [Paper](https://arxiv.org/abs/2505.18221)

<a id="ref91"></a>[91] "MEVER: Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval," *EACL*, 2026. [Paper](https://aclanthology.org/2026.eacl-long.242/)

<a id="ref92"></a>[92] Y. Zhang, D. She, B. Ji, Q. Geng, Z. Zhou, Y. Wang, "VMD-FACT: A New Video Dataset and MLLM-based Method for Detecting Realistic AI-Generated Video Misinformation (RAVM / IEEG)," *CVPR*, 2026. [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_VMD-FACT_A_New_Video_Dataset_and_MLLM-based_method_for_Detecting_CVPR_2026_paper.html)

<a id="ref93"></a>[93] Yang et al., "Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection (PCGR)," *CVPR*, 2026. [Paper](https://arxiv.org/abs/2603.25203) | [Code](https://github.com/2302Jerry/pcgr)

<a id="ref94"></a>[94] Wang et al., "Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective (RETSIMD)," *AAAI*, 2026. [Paper](https://arxiv.org/abs/2511.06284) | [Code](https://github.com/wangbing1416/RETSIMD)

<a id="ref95"></a>[95] Lin et al., "EvoGraph-R1: Self-Evolving Multimodal Knowledge Hypergraphs for Agentic Retrieval," *CVPR*, 2026. [Paper](https://arxiv.org/abs/2607.12764) | [Code](https://github.com/ninjaX2o/EvoGraph-R1)

<a id="ref96"></a>[96] S.-I. Papadopoulos, Z. Chrysidis, C. Koutlis, S. Papadopoulos, P. C. Petrantonakis, "Evidence Triangulation for Multimodal Fact-Checking in the Wild (TRENT / X-POSE)," *ECCV*, 2026. [Paper](https://arxiv.org/abs/2606.31367) | [Code](https://github.com/stevejpapad/evidence-triangulation)

<a id="ref97"></a>[97] "Novel Claim or Déjà Vu? Rethinking “Contamination-Free” Dynamic Evaluation for Multimodal Automated Fact-Checking," *arXiv:2607.23514*, 2026. [Paper](https://arxiv.org/abs/2607.23514)

<a id="ref98"></a>[98] "Overview of the AVerImaTeC Shared Task (FEVER 2026; winner VILLAIN)," *FEVER Workshop*, 2026. [Paper](https://aclanthology.org/2026.fever-1.6/) | [arXiv](https://arxiv.org/abs/2602.11221)

<a id="ref99"></a>[99] M. Rothermel, M. Kornmann, M. Rohrbach, A. Rohrbach, "VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking," *ACL*, 2026. [Paper](https://aclanthology.org/2026.acl-long.1948/) | [arXiv](https://arxiv.org/abs/2601.08611) | [Website](https://veritas.mai.informatik.tu-darmstadt.de/)

<a id="ref100"></a>[100] C. Xu, C. Jin, Y. Niu, N. Yan, Y. Mei, S. Guan, L. Chen, M-T. Kechadi, "LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection," *ACL*, 2026. [Paper](https://aclanthology.org/2026.acl-long.546/) | [Code](https://github.com/bebxy/livefact) | [arXiv](https://arxiv.org/abs/2604.04815)

<a id="ref101"></a>[101] S. Singh, I. Mujtahid, M.-Y. Kan, K. Jaidka, "CommunityFact: A Dynamic, Multilingual, Multi-domain Benchmark for Misinformation Detection in the Wild," *arXiv:2605.30241*, 2026. [Paper](https://arxiv.org/abs/2605.30241) | [Code](https://github.com/sahajps/CommunityFact) | [Dataset](https://huggingface.co/datasets/sahajps/CommunityFact)

<a id="ref102"></a>[102] V. Venktesh et al., "QuanTemp: A Real-world Open-domain Benchmark for Fact-checking Numerical Claims," *SIGIR*, 2024. [Paper](https://arxiv.org/abs/2403.17169) | [Code](https://github.com/factiverse/QuanTemp)

<a id="ref103"></a>[103] M. Strong, A. Vlachos, "TSVer: A Benchmark for Fact Verification Against Time-Series Evidence," *EMNLP*, 2025. [Paper](https://aclanthology.org/2025.emnlp-main.1519/)

<a id="ref104"></a>[104] J. Geng, J. Tonglet, I. Gurevych, "M4FC: a Multimodal, Multilingual, Multicultural, Multitask Real-World Fact-Checking Dataset," *arXiv:2510.23508*, 2025. [Paper](https://arxiv.org/abs/2510.23508) | [Code](https://github.com/UKPLab/M4FC)

<a id="ref105"></a>[105] W. Xu et al., "MMM-Fact: A Multimodal, Multi-Domain Fact-Checking Dataset with Multi-Level Retrieval Difficulty," *arXiv:2510.25120*, 2025. [Paper](https://arxiv.org/abs/2510.25120) | [Dataset](https://huggingface.co/datasets/Wenyan0110/MMM-Fact)

<a id="ref106"></a>[106] R. Cao et al., "AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web," *NeurIPS Datasets & Benchmarks*, 2025. [Paper](https://arxiv.org/abs/2505.17978) | [Dataset](https://huggingface.co/datasets/Rui4416/AVerImaTeC) | [Website](https://fever.ai/dataset/averimatec.html)

<a id="ref107"></a>[107] J. Tonglet, M.-F. Moens, I. Gurevych, "“Image, Tell me your story!” Predicting the Original Meta-Context of Visual Misinformation (5Pils)," *EMNLP*, 2024. [Paper](https://aclanthology.org/2024.emnlp-main.448/) | [Code](https://github.com/UKPLab/5pils)

<a id="ref108"></a>[108] J. Tonglet et al., "COVE: COntext and VEracity Prediction for Out-of-Context Images," *NAACL*, 2025. [Paper](https://aclanthology.org/2025.naacl-long.102/) | [Code](https://github.com/UKPLab/naacl2025-cove)

<a id="ref109"></a>[109] B. M. Yao et al., "End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models (MOCHEG)," *SIGIR*, 2023. [Paper](https://arxiv.org/abs/2205.12487) | [Code](https://github.com/VT-NLP/Mocheg)

<a id="ref110"></a>[110] S.-I. Papadopoulos et al., "VERITE: A Robust Benchmark for Multimodal Misinformation Detection Accounting for Unimodal Bias," *IJMIR*, 2024. [Paper](https://arxiv.org/abs/2304.14133) | [Code](https://github.com/stevejpapad/image-text-verification)

<a id="ref111"></a>[111] M. Pikuliak et al., "Multilingual Previously Fact-Checked Claim Retrieval (MultiClaim)," *EMNLP*, 2023. [Paper](https://aclanthology.org/2023.emnlp-main.1027/) | [Code](https://github.com/kinit-sk/multiclaim) | [Dataset](https://zenodo.org/records/7737983)

<a id="ref112"></a>[112] S. Suryavardan et al., "Factify 2: A Multimodal Fake News and Satire News Dataset," *De-Factify Workshop @ AAAI*, 2023. [Paper](https://arxiv.org/abs/2304.03897)

<a id="ref113"></a>[113] X. Hu, Z. Guo, J. Chen, L. Wen, P. S. Yu, "MR2: A Benchmark for Multimodal Retrieval-Augmented Rumor Detection in Social Media," *SIGIR*, 2023. [Paper](https://dl.acm.org/doi/10.1145/3539618.3591896) | [Code](https://github.com/THU-BPM/MR2)

<a id="ref114"></a>[114] G. Luo, T. Darrell, A. Rohrbach, "NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media," *EMNLP*, 2021. [Paper](https://arxiv.org/abs/2104.05893) | [Code](https://github.com/g-luo/news_clippings)

<a id="ref115"></a>[115] S. Aneja, C. Bregler, M. Nießner, "COSMOS: Catching Out-of-Context Image Misuse Using Self-Supervised Learning," *AAAI*, 2023. [Paper](https://arxiv.org/abs/2101.06278) | [Code](https://github.com/shivangi-aneja/COSMOS)

<a id="ref116"></a>[116] G. Biamby et al., "Twitter-COMMs: Detecting Climate, COVID, and Military Multimodal Misinformation," *NAACL*, 2022. [Paper](https://arxiv.org/abs/2112.08594)

<a id="ref117"></a>[117] "XFacta: Contemporary, Real-World Dataset and Evaluation for Multimodal Misinformation Detection with Multimodal LLMs," *arXiv:2508.09999*, 2025. [Paper](https://arxiv.org/abs/2508.09999) | [Code](https://github.com/neu-vi/XFacta)

<a id="ref118"></a>[118] K. Shu et al., "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media," *Big Data 8(3)*, 2020. [Paper](https://arxiv.org/abs/1809.01286) | [Code](https://github.com/KaiDMML/FakeNewsNet)

<a id="ref119"></a>[119] D. S. Nielsen, R. McConville, "MuMiN: A Large-Scale Multilingual Multimodal Fact-Checked Misinformation Social Network Dataset," *SIGIR*, 2022. [Paper](https://arxiv.org/abs/2202.11684) | [Code](https://github.com/MuMiN-dataset/mumin-build) | [Website](https://mumin-dataset.github.io/)

<a id="ref120"></a>[120] M. Gruppi, B. D. Horne, S. Adalı, "NELA-GT-2022: A Large Multi-Labelled News Dataset for the Study of Misinformation in News Articles," *arXiv:2203.05659*, 2022. [Paper](https://arxiv.org/abs/2203.05659) | [Code](https://github.com/MELALab/nela-gt)

<a id="ref121"></a>[121] Z. Zhou et al., "FineFake: A Knowledge-Enriched Dataset for Fine-Grained Multi-Domain Fake News Detection," *Information Fusion (arXiv:2404.01336)*, 2026. [Paper](https://arxiv.org/abs/2404.01336) | [Code](https://github.com/Accuser907/FineFake)

<a id="ref122"></a>[122] L. Cui, D. Lee, "CoAID: COVID-19 Healthcare Misinformation Dataset," *arXiv:2006.00885*, 2020. [Paper](https://arxiv.org/abs/2006.00885) | [Code](https://github.com/cuilimeng/CoAID)

<a id="ref123"></a>[123] X. Zhou et al., "ReCOVery: A Multimodal Repository for COVID-19 News Credibility Research," *CIKM*, 2020. [Paper](https://arxiv.org/abs/2006.05557) | [Code](https://github.com/apurvamulay/ReCOVery)

<a id="ref124"></a>[124] Y. Li et al., "MM-COVID: A Multilingual and Multimodal Data Repository for Combating COVID-19 Disinformation," *arXiv:2011.04088*, 2020. [Paper](https://arxiv.org/abs/2011.04088)

<a id="ref125"></a>[125] Q. Nan et al., "MDFEND: Multi-domain Fake News Detection (Weibo21)," *CIKM*, 2021. [Paper](https://arxiv.org/abs/2201.00987) | [Code](https://github.com/kennqiang/MDFEND-Weibo21)

<a id="ref126"></a>[126] Y. Li, H. He, J. Bai, D. Wen, "MCFEND: A Multi-source Benchmark Dataset for Chinese Fake News Detection," *WWW*, 2024. [Paper](https://arxiv.org/abs/2403.09092) | [Code](https://github.com/TrustworthyComp/MCFEND)

<a id="ref127"></a>[127] J. Ma, W. Gao, K.-F. Wong, "Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning (Twitter15/16)," *ACL*, 2017. [Paper](https://aclanthology.org/P17-1066/)

<a id="ref128"></a>[128] A. Zubiaga et al., "Analysing How People Orient to and Spread Rumours in Social Media by Looking at Conversational Threads (PHEME)," *PLOS ONE*, 2016. [Paper](https://doi.org/10.1371/journal.pone.0150989) | [Datasets](https://www.zubiaga.org/datasets/)

<a id="ref129"></a>[129] J. Ma et al., "Detecting Rumors from Microblogs with Recurrent Neural Networks (Weibo rumour dataset)," *IJCAI*, 2016. [Paper](https://www.ijcai.org/Proceedings/16/Papers/537.pdf)

<a id="ref130"></a>[130] K. Nakamura, S. Levy, W. Y. Wang, "r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection," *LREC*, 2020. [Paper](https://arxiv.org/abs/1911.03854) | [Code](https://github.com/entitize/Fakeddit)

<a id="ref131"></a>[131] Z. Jin et al., "Multimodal Fusion with Recurrent Neural Networks for Rumor Detection on Microblogs (Weibo multimodal)," *ACM MM*, 2017. [Paper](https://doi.org/10.1145/3123266.3123454)

<a id="ref132"></a>[132] C. Boididou et al., "Verifying Multimedia Use at MediaEval 2016 (Image Verification Corpus)," *MediaEval Workshop*, 2016. [Code](https://github.com/MKLab-ITI/image-verification-corpus)

<a id="ref133"></a>[133] Wang et al., "FMNV: A Dataset of Media-Published News Videos for Fake News Detection," *Springer LNCS (arXiv:2504.07687)*, 2025. [Paper](https://arxiv.org/abs/2504.07687) | [Code](https://github.com/DennisIW/FMNV)

<a id="ref134"></a>[134] O. Papadopoulou et al., "A Corpus of Debunked and Verified User-Generated Videos (Fake Video Corpus)," *Online Information Review*, 2019. [Paper](https://doi.org/10.1108/OIR-03-2018-0101) | [Code](https://github.com/MKLab-ITI/fake-video-corpus)

<a id="ref135"></a>[135] F. Liu et al., "COVID-VTS: Fact Extraction and Verification on Short Video Platforms," *EACL*, 2023. [Paper](https://aclanthology.org/2023.eacl-main.14/) | [Code](https://github.com/FuxiaoLiu/Twitter-Video-dataset) | [arXiv](https://arxiv.org/abs/2302.07919)

<a id="ref136"></a>[136] J. Thorne et al., "FEVER: a Large-scale Dataset for Fact Extraction and VERification," *NAACL*, 2018. [Paper](https://arxiv.org/abs/1803.05355) | [Dataset](https://fever.ai/dataset/fever.html)

<a id="ref137"></a>[137] M. Schlichtkrull, Z. Guo, A. Vlachos, "AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web," *NeurIPS Datasets & Benchmarks*, 2023. [Paper](https://arxiv.org/abs/2305.13117) | [Dataset](https://fever.ai/dataset/averitec.html)

<a id="ref138"></a>[138] W. Y. Wang, "“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection (LIAR)," *ACL*, 2017. [Paper](https://arxiv.org/abs/1705.00648)

<a id="ref139"></a>[139] I. Augenstein et al., "MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims," *EMNLP*, 2019. [Paper](https://arxiv.org/abs/1909.03242)

<a id="ref140"></a>[140] A. Gupta, V. Srikumar, "X-FACT: A New Benchmark Dataset for Multilingual Fact Checking," *ACL*, 2021. [Paper](https://arxiv.org/abs/2106.09248) | [Code](https://github.com/utahnlp/x-fact)

<a id="ref141"></a>[141] A. Hanselowski et al., "A Richly Annotated Corpus for Different Tasks in Automated Fact-Checking (Snopes corpus)," *CoNLL*, 2019. [Paper](https://arxiv.org/abs/1911.01214)

<a id="ref142"></a>[142] R. Baly et al., "Predicting Factuality of Reporting and Bias of News Media Sources," *EMNLP*, 2018. [Paper](https://arxiv.org/abs/1810.01765) | [Code](https://github.com/ramybaly/News-Media-Reliability)

<a id="ref143"></a>[143] H. Lin et al., "High Level of Correspondence Across Different News Domain Quality Rating Sets," *PNAS Nexus*, 2023. [Paper](https://academic.oup.com/pnasnexus/article/2/9/pgad286/7258994) | [Code](https://github.com/hauselin/domain-quality-ratings)

<a id="ref144"></a>[144] C. Thibault et al., "A Guide to Misinformation Detection Datasets," *arXiv:2411.05060*, 2024. [Paper](https://arxiv.org/abs/2411.05060) | [Website](https://misinfo-datasets.complexdatalab.com/)

<a id="ref145"></a>[145] S. Huang et al., "Temporal Graph Benchmark for Machine Learning on Temporal Graphs (TGB)," *NeurIPS Datasets & Benchmarks*, 2023. [Paper](https://arxiv.org/abs/2307.01026) | [Code](https://github.com/shenyangHuang/TGB) | [Website](https://tgb.complexdatalab.com/)

<a id="ref146"></a>[146] J. Gastinger et al., "TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs," *NeurIPS Datasets & Benchmarks*, 2024. [Paper](https://arxiv.org/abs/2406.09639)

<a id="ref147"></a>[147] J. Zhang et al., "DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs," *NeurIPS Datasets & Benchmarks*, 2024. [Paper](https://arxiv.org/abs/2406.12072) | [Code](https://github.com/zjs123/DTGB)

<a id="ref148"></a>[148] J. Peng et al., "GDGB: A Benchmark for Generative Dynamic Text-Attributed Graph Learning," *ICLR*, 2026. [Paper](https://arxiv.org/abs/2507.03267)

<a id="ref149"></a>[149] B. Rozemberczki et al., "PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models," *CIKM*, 2021. [Paper](https://arxiv.org/abs/2104.07788) | [Code](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)

<a id="ref150"></a>[150] Y. Zheng, L. Yi, Z. Wei, "A Survey of Dynamic Graph Neural Networks," *Frontiers of Computer Science*, 2025. [Paper](https://arxiv.org/abs/2404.18211)

<a id="ref151"></a>[151] Z. Guo, M. Schlichtkrull, A. Vlachos, "A Survey on Automated Fact-Checking," *TACL*, 2022. [Paper](https://arxiv.org/abs/2108.11896)

<a id="ref152"></a>[152] M. Akhtar et al., "Multimodal Automated Fact-Checking: A Survey," *Findings of EMNLP*, 2023. [Paper](https://arxiv.org/abs/2305.13507)

<a id="ref153"></a>[153] "Fake News Detection through Graph-based Neural Networks: A Survey," *arXiv:2307.12639*, 2023. [Paper](https://arxiv.org/abs/2307.12639)

<a id="ref154"></a>[154] P. Qi et al., "Combating Online Misinformation Videos: Characterization, Detection, and Future Directions," *ACM MM*, 2023. [Paper](https://arxiv.org/abs/2302.03242) | [Code](https://github.com/ICTMCG/Awesome-Misinfo-Video-Detection)


---

## 🤝 Contributing
Contributions are welcome! Please open an issue or pull request to add papers, datasets or implementations. When adding an entry, link the primary source (proceedings, arXiv, or official repository) and state the venue only if it is confirmed.

---

## ⭐ Acknowledgements
If you find this list helpful, please star ⭐ the repository to support the project.
