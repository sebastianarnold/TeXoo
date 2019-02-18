# TeXoo – Topic Classification and Segmentation (SECTOR)

Annotators for **SECTOR** models from WikiSection dataset.

| Package / Class                               | Description / Reference                                                |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| [SectorAnnotator](texoo-sector/src/main/java/de/datexis/sector/SectorAnnotator.java)      | Topic Segmentation and Classification for Long Documents               |

### Command Line Usage:

- run ```bin/run-docker texoo-train-sector```

```
usage: texoo-train-sector -i <arg> -o <arg> [-u]
TeXoo: train SectorAnnotator from WikiSection dataset
 -i,--input <arg>    file name of WikiSection training dataset
 -o,--output <arg>   path to create and store the model
 -u,--ui             enable training UI (http://127.0.0.1:9000)

```

### Cite

If you use this module for research, please cite:

> Sebastian Arnold, Rudolf Schneider, Philippe Cudré-Mauroux, Felix A. Gers and Alexander Löser. "SECTOR: A Neural Model for Coherent Topic Segmentation and Classification." Transactions of the Association for Computational Linguistics (2019). <https://arxiv.org/abs/1902.04793>
