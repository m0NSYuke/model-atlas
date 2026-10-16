# Interface references and decisions

The architecture maps and their paper-grounded connections are documented separately in [ARCHITECTURE_SOURCES.md](ARCHITECTURE_SOURCES.md). This file records interface patterns used to make those maps easier to read.

| Reference | Observed pattern | Atlas implementation |
| --- | --- | --- |
| [TensorBoard Graphs](https://www.tensorflow.org/tensorboard/graphs) | A conceptual graph supports structural reading; selecting a node reveals inputs, outputs, and shapes. | A dedicated diagram view shows directed edges, while the Inspector lists clickable incoming and outgoing connections. |
| [CNN Explainer](https://poloclub.github.io/papers/20-vis-cnnexplainer.pdf) | Coordinated overview and detail help learners move from an entire network to a particular operation. | The fitted diagram gives an overview; 1:1, Trace, Details, and the Inspector reveal local structure and tensor shapes. |
| [Transformer Explainer](https://github.com/poloclub/transformer-explainer) | Interactive selection makes an abstract architecture inspectable step by step. | Selecting a node in the tree, diagram, or 3D view updates the same selection and explanation. |
| [Netron](https://github.com/lutzroeder/netron) | A model graph and a details pane serve different reading tasks. | The graph prioritizes topology; the Inspector carries parameters, shapes, source references, and learning notes. |

## Visual rules

- Use explicit arrowheads. Solid cyan edges represent the main data path. Dashed edges and distinct colors mark conditions, training relationships, feedback, and reused networks.
- Keep graph positions deterministic in Model IR. Route long bypasses through `edge.via` where a direct line would obscure the main lane.
- Draw diagrams as readable cards on a scrollable board; use Fit for overview and 1:1 for reading. The 3D view remains available for spatial exploration.
- Trace only direct incoming and outgoing edges from the selected node. Dim the rest, without changing the graph data.
- Show full node metadata in the Inspector instead of placing every label and shape on the overview.

The references informed interaction and visual hierarchy. Architecture connections are based on the cited model papers and the Atlas Model IR, rather than inferred from these interface references.
