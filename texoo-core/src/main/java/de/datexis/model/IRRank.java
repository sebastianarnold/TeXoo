package de.datexis.model;

import com.google.common.base.Objects;

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

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof IRRank)) {
      return false;
    }
    IRRank irRank = (IRRank) o;
    return rank == irRank.rank;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(rank);
  }

  @Override
  public String toString() {
    return "IRRank [rank=" + rank + "]";
  }
}
