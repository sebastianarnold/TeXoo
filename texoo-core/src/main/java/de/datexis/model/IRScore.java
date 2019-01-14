package de.datexis.model;

import com.google.common.base.Objects;

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

  @Override
  public String toString() {
    return "IRScore [score=" + score + "]";
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof IRScore)) {
      return false;
    }
    IRScore irScore = (IRScore) o;
    return Double.compare(irScore.score, score) == 0;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(score);
  }
}
