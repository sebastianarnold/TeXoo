# TeXoo – A Zoo of Text Extractors

TeXoo is an tagging framework developed at DATEXIS, Beuth University of Applied Sciences Berlin. TeXoo comes with a NLP-style document model and a zoo of Deep Learning extraction models which you can access in texoo-models module. Here is a brief overview:


## Modules in TeXoo

### **texoo-core** – Document Model and Core Library

| Package / Class                             | Description                                   |
| ------------------------------------------- | --------------------------------------------- |
| **de.datexis.model**.*                      | TeXoo Document model (see below)              |
| **de.datexis.encoder**.*                    | Implementations of Bag-of-words, Word2Vec, Trigrams, etc. |
| de.datexis.preprocess.**DocumentFactory**   | Factory to create Document objects from text  |
| de.datexis.annotator.**AnnotatorFactory**   | Factory to create and load models from the zoo |
| de.datexis.common.**ObjectSerializer**      | Helper methods to import/export JSON          |
	
### **texoo-entity-recognition** NER implementation using Deeplearning4j

| Package / Class                               | Description / Reference                                                |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| de.datexis.ner.**GenericMentionAnnotator** | Robust Named Entity Recognition (NER) with pre-trained models for English and German <http://arxiv.org/abs/1608.06757> |

### **texoo-entity-linking** NEL implementation using Deeplearning4j

Training functions Named Entity Linking models from various datasets (currently under development)

| Package / Class                               | Description / Reference                                                |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| de.datexis.nel.**NamedEntityAnnotator**    | Named Entity Linking used in TASTY <https://www.aclweb.org/anthology/C/C16/C16-2024.pdf> |
| de.datexis.index.**ArticleIndexFactory**   | Knowledge Base implemented as local Lucene Index which imports Wikidata entities |

### **texoo-sector** – topic classification and text segmentation using LSTM

Training functions SECTOR models from WikiSection dataset (currently under development)

| Package / Class                               | Description / Reference                                                |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| de.datexis.sector.**SectorAnnotator**      | Topic Segmentation and Classification for Long Documents               |



### **texoo-examples** – Examples to Start your Implementation


## Installation and Usage

### Prerequisites

- **Oracle Java 8**
- **Apache Maven** Build system for Java  
<https://maven.apache.org/guides/index.html>

### Installation

- [optional] Create a configuration file and adapt your local model paths:  
`vim texoo-core/src/main/resources/texoo.properties`
- Compile, test and install Texoo:  
`cd texoo-core && mvn install`

### Usage

- Run the example:  
`cd texoo-core && mvn exec:java -Dexec.mainClass=de.datexis.examples.AnnotateEntityRecognition`

## Documentation

### Frameworks used in TeXoo

- **Deeplearning4j** Machine learning library  
<http://deeplearning4j.org/documentation>
- **ND4J** Scientific computing library  
<http://nd4j.org/userguide>
- **Stanford CoreNLP** Natural language processing  
<http://stanfordnlp.github.io/CoreNLP/>

### TeXoo Data Model

<p align="center"><img src="documentation/texoo_model_document.png" width="80%"></p>

### References

If you use this work, please cite:

Sebastian Arnold, Felix A. Gers, Torsten Kilias, Alexander Löser: Robust Named Entity Recognition in Idiosyncratic Domains. arXiv:1608.06757 [cs.CL] 2016

## License

Until now, TeXoo is for internal use only.
