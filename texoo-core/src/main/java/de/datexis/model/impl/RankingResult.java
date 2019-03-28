package de.datexis.model.impl;

import de.datexis.model.Document;
import de.datexis.model.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Result that is extended to hold a ranking-based result (e.g. 1 is the best result, 2 is second best, ...)
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class RankingResult extends Result/*<Integer>*/ {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected Integer rank = null;
  
  public RankingResult(Source source) {
    super(source);
    this.setSortDescending(false);
  }
  
  public RankingResult(Source source, Document doc, int begin, int end) {
    super(source, doc, begin, end);
  }
  
  /**
   * Set the ranking position 1..N
   */
  public void setRank(Integer rank) {
    this.rank = rank;
  }
  
  /**
   * @return the ranking position 1..N
   */
  public Integer getRank() {
    return rank;
  }
  
}
