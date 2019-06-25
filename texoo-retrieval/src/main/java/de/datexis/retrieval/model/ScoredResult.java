package de.datexis.retrieval.model;

import de.datexis.model.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Result that is holds a score where 1.0 is the best score and 0.0 the worst.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ScoredResult extends RelevanceResult {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected boolean isRelevant = false;
  
  /** default constructor for JSON deserialization */
  protected ScoredResult() {};
  
  public ScoredResult(Source source) {
    super(source);
  }
  
  public ScoredResult(Source source, Document doc, int begin, int end) {
    super(source, doc, begin, end);
  }
  
  public ScoredResult(Source source, Document doc, double score) {
    super(source);
    this.setDocumentRef(doc);
    this.setScore(score);
  }
  
  /**
   * Assign if this result was found relevant during evaluation
   */
  public void setRelevant(boolean relevant) {
    isRelevant = relevant;
  }
  
  /**
   * @return TRUE iff the result was found relevant during evaluation
   */
  @Override
  public boolean isRelevant() {
    return isRelevant;
  }
  
}
