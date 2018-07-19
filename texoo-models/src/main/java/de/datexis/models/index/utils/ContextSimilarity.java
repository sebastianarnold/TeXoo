package de.datexis.models.index.utils;

import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.search.similarities.ClassicSimilarity;

/**
 * Term similarity without normalization for Lucene.
 * @author sarnold
 */
public class ContextSimilarity extends ClassicSimilarity {
  
  /**
   * Compute an index-time normalization value for this field instance.
   * <p>
   * This value will be stored in a single byte lossy representation by 
   * {@link #encodeNormValue(float)}.
   * 
   * @param state statistics of the current field (such as length, boost, etc)
   * @return an index-time normalization value
   */
  @Override
  public float lengthNorm(FieldInvertState state) {
    return state.getBoost() * 1.0f;
  }
  
  /** Computes the normalization value for a query given the sum of the squared
   * weights of each of the query terms.  This value is multiplied into the
   * weight of each query term. While the classic query normalization factor is
   * computed as 1/sqrt(sumOfSquaredWeights), other implementations might
   * completely ignore sumOfSquaredWeights (ie return 1).
   *
   * <p>This does not affect ranking, but the default implementation does make scores
   * from different queries more comparable than they would be by eliminating the
   * magnitude of the Query vector as a factor in the score.
   *
   * @param sumOfSquaredWeights the sum of the squares of query term weights
   * @return a normalization factor for query weights
   */
  @Override
  public float queryNorm(float sumOfSquaredWeights) {
    return 1.0f; //(float)(1.0 / Math.sqrt(sumOfSquaredWeights));
  }
  
  /** Computes a score factor based on the fraction of all query terms that a
   * document contains.  This value is multiplied into scores.
   *
   * <p>The presence of a large portion of the query terms indicates a better
   * match with the query, so implementations of this method usually return
   * larger values when the ratio between these parameters is large and smaller
   * values when the ratio between them is small.
   *
   * @param overlap the number of query terms matched in the document
   * @param maxOverlap the total number of terms in the query
   * @return a score factor based on term overlap with the query
   */
  @Override
  public float coord(int overlap, int maxOverlap) {
    return 1.f;
  }
  
}
