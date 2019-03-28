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
    this.setSortDescending(false);
  }
  
  public ScoredResult(Source source, Document doc, int begin, int end) {
    super(source, doc, begin, end);
  }
  
}
