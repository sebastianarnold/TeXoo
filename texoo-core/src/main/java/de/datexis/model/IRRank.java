package de.datexis.model;

/**
 * Represents the relevance of an document by rank
 */
public class IRRank extends IRRelevance {
  /**
   * The relevance rank of an document
   */
  private int rank;

  public IRRank(int rank) {
    this.rank = rank;
  }

  public int getRank() {
    return rank;
  }
}
