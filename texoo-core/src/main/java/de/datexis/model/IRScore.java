package de.datexis.model;

/**
 * Represents the relevance of an document by score
 */
public class IRScore extends IRRelevance {
  /**
   * The relevance score of an document
   */
  private double score;

  public IRScore(double score) {
    this.score = score;
  }

  public double getScore() {
    return score;
  }
}
