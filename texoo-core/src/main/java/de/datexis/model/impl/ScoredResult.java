package de.datexis.model.impl;

import de.datexis.model.Document;
import de.datexis.model.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Result that is holds a score where 1.0 is the best score and 0.0 the worst.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ScoredResult extends Result {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  public ScoredResult(Source source) {
    super(source);
    this.setSortDescending(true);
  }
  
  public ScoredResult(Source source, Document doc, int begin, int end) {
    super(source, doc, begin, end);
    this.setSortDescending(true);
  }
  
  public ScoredResult(Source source, Document doc, double score) {
    super(source);
    this.setDocumentRef(doc);
    this.setScore(score);
    this.setSortDescending(true);
  }
  
  /**
   * @return 1, because a scored result is always assumed to be relevant
   */
  @Override
  public Integer getRelevance() {
    return 1;
  }
  
  /**
   * @return true, because a scored result is always assumed to be relevant
   */
  @Override
  public boolean isRelevant() {
    return true;
  }
  
}
