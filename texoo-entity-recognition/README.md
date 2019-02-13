# TeXoo – Named Entity Recognition (NER)

This module contains Annotators for **Named Entity Recognition (NER)**. This is a very robust deep learning model that can be trained with only 4000-5000 sentences. It is based on a bidirection LSTM with Letter-trigram encoding, see <http://arxiv.org/abs/1608.06757>.

### Command Line Usage:

- run ```bin/run-docker texoo-annotate-ner```

```
usage: texoo-annotate-ner -i <arg> [-o <arg>]
TeXoo: run pre-trained MentionAnnotator model
 -i,--input <arg>    path or file name for raw input text
 -o,--output <arg>   path to create and store the output JSON, otherwise dump to stdout
```

- run ```bin/run-docker texoo-train-ner```

```
usage: texoo-train-ner -i <arg> [-l <arg>] -o <arg> [-t <arg>] [-u] [-v
       <arg>]
TeXoo: train MentionAnnotator with CoNLL annotations
 -i,--input <arg>        path to input training data (CoNLL format)
 -l,--language <arg>     language to use for sentence splitting and
                         stopwords (EN or DE)
 -o,--output <arg>       path to create and store the model
 -t,--test <arg>         path to test data (CoNLL format)
 -u,--ui                 enable training UI (http://127.0.0.1:9000)
 -v,--validation <arg>   path to validation data (CoNLL format)
```

- run ```bin/run-docker texoo-train-ner-seed```

```
usage: texoo-train-ner-seed -i <arg> -o <arg> -s <arg> [-u]
TeXoo: train MentionAnnotator with seed list
 -i,--input <arg>    path and file name pattern for raw input text
 -o,--output <arg>   path to create and store the model
 -s,--seed <arg>     path to seed list text file
 -u,--ui             enable training UI (http://127.0.0.1:9000)
```

### Java Classes:

| Package / Class                               | Description / Reference                                                |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| [MentionAnnotator](texoo-entity-recognition/src/main/java/de/datexis/ner/MentionAnnotator.java)    | Named Entity Recognition |
| [GenericMentionAnnotator](texoo-entity-recognition/src/main/java/de/datexis/ner/GenericMentionAnnotator.java)   | Pre-trained models for English and German |
| [MatchingAnnotator](texoo-entity-recognition/src/main/java/de/datexis/ner/MatchingAnnotator.java)    | Gazetteer that uses Lists to annotate Documents |
| [CoNLLDatasetReader](texoo-entity-recognition/src/main/java/de/datexis/ner/reader/CoNLLDatasetReader.java) | Reader for CoNLL files |

### Cite

If you use this module for research, please cite:

> Sebastian Arnold, Felix A. Gers, Torsten Kilias, Alexander Löser: Robust Named Entity Recognition in Idiosyncratic Domains. arXiv:1608.06757 [cs.CL] 2016
