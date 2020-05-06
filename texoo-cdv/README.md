# CDV – Contextualized Discourse Vectors

Code and data for:

**Learning Contextualized Document Representations for Healthcare Answer Retrieval.** Sebastian Arnold, Betty van Aken, Paul Grundmann, Felix A. Gers and Alexander Löser. The Web Conference 2020.

This code is based on the [TeXoo Framework](https://github.com/sebastianarnold/texoo) and [Eclipse Deeplearning4j](https://github.com/eclipse/deeplearning4j).

## Preparing datasets

We provide full training and evaluation data from Wikipedia (WikiSectionQA) and annotations that extend MedQuAD and HealthQA datasets. You have to run `convert.sh` inside the datasets' directories in order to prepare the json files from the original sources. Please see the instructions in the [data directory](data) for more details.

## Training a model

- run ```bin/train-cdv```

```
TeXoo: train contextualized discourse vectors (CDV)
 -a,--aspect <arg>        path to the aspect embedding
 -b,--balancing           use class balancing during training
 -d,--datasetname <arg>   name of the data set, e.g. wd_disease
 -e,--entity <arg>        path to the entity embedding
 -i,--dataset <arg>       path to the WikiSection training dataset
 -m,--modelname <arg>     model name
 -o,--output <arg>        path to create the output folder in
 -s,--search <arg>        search path for pre-trained word embeddings
 -u,--ui                  enable training UI
 -w,--wordemb <arg>       path to a pretrained word embedding
```

- Examples:

```
bin/train-cdv -m "mymodel" -i data/Train/wd_disease_train.json -d wd_disease -s models/common -w models/common/en_disease_skipgram.bin -e models/ENC-E@wd_disease+ft-lstm+128 -a models/ENC-A@wd_disease+ft-lstm+128 -o models
```

Hyperparameters are configured in the source file. See below how to get access to pre-trained `ENC-E` / `ENC-A` embeddings.

## Running the evaluation

- run ```bin/evaluate-cdv```

```
TeXoo: evaluate CDV answer retrieval
 -a,--aspect <arg>    optional path to a CDV single-task aspect model
 -d,--dataset <arg>   path to the evaluation dataset (json)
 -e,--entity <arg>    optional path to a CDV single-task entity model
 -m,--model <arg>     path to the pre-trained CDV multi-task model
 -p,--path <arg>      search path to sentence embedding models (if not
                      provided by the model itself)
```

- Example:

```
bin/evaluate-cdv -m models/CDV@wd_disease+avg-fasttext -p models/common -d data/WikiSectionQA/WikiSectionQA_test.json

```

Please contact sarnold(at)beuth-hochschule(dot)de to get access to the pre-trained `CDV+avg-fasttext` model.

## Cite

```
@inproceedings{arnold2020learning,
  author = {Arnold, Sebastian and {van Aken}, Betty and Grundmann, Paul and Gers, Felix A. and L{\"o}ser, Alexander},
  title = {Learning {{Contextualized Document Representations}} for {{Healthcare Answer Retrieval}}},
  booktitle = {Proceedings of The Web Conference 2020 (WWW '20)},
  year = {2020},
  doi = {10.1145/3366423.3380208}
}
```
