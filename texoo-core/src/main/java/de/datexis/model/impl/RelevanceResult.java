package de.datexis.model.impl;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import de.datexis.model.Document;
import de.datexis.model.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Result that is extended to hold Cranfield-style relevance judgements (0 - not relevant, 1 - relevant, or more ...)
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class RelevanceResult extends Result {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected Integer relevance = null;
  
  /** default constructor for JSON deserialization */
  protected RelevanceResult() {};
  
  public RelevanceResult(Source source) {
    super(source);
  }
  
  public RelevanceResult(Source source, Document doc, int begin, int end) {
    super(source, doc, begin, end);
  }
  
  public RelevanceResult(Source source, Document doc, int relevance) {
    super(source);
    this.setDocumentRef(doc);
    this.setRelevance(relevance);
  }
  
  /**
   * Set a relevance judgement: 0 - not relevant, 1 - relevant.
   * Alternatively you can also use 1 - partly relevant, 2 - very relevant.
   */
  public void setRelevance(Integer relevance) {
    this.relevance = relevance;
  }
  
  /**
   * @return a relevance judgement: 0 - not relevant, 1 - relevant.
   * Alternatively you can also use 1 - partly relevant, 2 - very relevant.
   */
  public Integer getRelevance() {
    return relevance;
  }
  
  /**
   * @return True, if the result was labeled as relevant (e.g. relevance > 0
   */
  @JsonIgnore
  public boolean isRelevant() {
    return getRelevance() > 0;
  }
  
}
