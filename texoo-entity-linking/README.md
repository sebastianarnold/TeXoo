# TeXoo – Named Entity Linking (NEL)

This module contains the Annotators for **Named Entity Linking (NEL)** (currently under development). There is no model included, but you can use the Knowledge Base and Annotators with your own datasets, see <https://www.aclweb.org/anthology/C/C16/C16-2024.pdf>.

| Package / Class                               | Description / Reference                                                |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| [NamedEntityAnnotator](texoo-entity-linking/src/main/java/de/datexis/nel/NamedEntityAnnotator.java)    | Named Entity Linking used in TASTY |
| [ArticleIndexFactory](texoo-entity-linking/src/main/java/de/datexis/index/ArticleIndexFactory.java)   | Knowledge Base implemented as local Lucene Index which imports Wikidata entities |

If you use this module for research, please cite:

> Sebastian Arnold, Robert Dziuba, Alexander Löser: TASTY: Interactive Entity Linking As-You-Type. COLING (Demos) 2016: 111–115