package de.datexis.retrieval.index;

import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public interface IVectorIndex {

  /**
   * @return k nearest keys for a given embedding.
   */
  List<IndexEntry> find(INDArray vec, int k);

  /**
   * @return nearest key for a given {0...1}^K dense vector embedding.
   */
  IndexEntry find(INDArray vec);

  /**
   * @return nearest key for a given {0...1}^K dense vector embedding.
   */
  default String findKey(INDArray vec) {
    return find(vec).key;
  }

  /**
   * @return nearest key for a given {0...1}^K dense vector embedding.
   */
  default int findIndex(INDArray vec) {
    IndexEntry result = find(vec);
    return result != null ? find(vec).index : -1;
  }

  /**
   * @return k nearest keys for a given embedding.
   */
  default List<String> findKeys(INDArray vec, int k) {
    return find(vec, k)
      .stream()
      .map(e -> e.key)
      .collect(Collectors.toList());
  }

  class IndexEntry implements Comparable<IndexEntry> {

    public int index;
    public String key;
    public double similarity;

    @Override
    public int compareTo(@NotNull IndexEntry o) {
      return Double.compare(similarity, o.similarity);
    }

    public String toString() {
      return String.format(Locale.ROOT, "%s (%.2f)", key, similarity);
    }
    
  }

}
