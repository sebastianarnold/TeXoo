package de.datexis.retrieval.index;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A Vocab holds a vocabulary of keys and maps it to INDArray vectors.
 * E.g. word embeddings, entity knowledge base
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public interface IVocabulary {

  /**
   * @return the {0...1}^K dense vector embedding for a given key.
   */
  INDArray lookup(String key);

  /**
   * @return the {0...1}^K dense vector embedding for a given index.
   */
  default INDArray lookup(int index) {
    return lookup(key(index));
  }

  /**
   * @return index of given key, -1 if not found
   */
  int index(String key);

  /**
   * @return key for given index
   */
  String key(int index);

  /**
   * @return number of keys N in the vocab
   */
  int size();

}
