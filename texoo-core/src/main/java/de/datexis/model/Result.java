package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.Comparator;

/**
 * A Result contains the retrieved Document and an Annotation that points to the
 * Span of the Result (e.g. a Paragraph)
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public abstract class Result/*<S extends Comparable> */extends Annotation {
  
//  protected static Comparator<Double> nullsafeDoubleComparator =
//  Comparator.nullsFirst(Double::compare);
  
  protected static Comparator<Result> resultComparator =
    Comparator.comparing(Result::getScore, Comparator.nullsLast(Double::compare))
      .thenComparing(Result::getRelevance, Comparator.nullsLast(Integer::compare));
  
  protected /*S*/ Double score = null;
  protected boolean sortDescending = true;
  protected String id = null;
  
  /** default constructor for JSON deserialization */
  protected Result() {};
  
  public Result(Source source) {
    super(source, "");
  }
  
  public Result(Source source, Document doc, int begin, int end) {
    super(source, "");
    this.setDocumentRef(doc);
    this.setBegin(begin);
    this.setEnd(end);
  }
  
  public void setId(String id) {
    this.id = id;
  }
  
  public String getId() {
    return this.id;
  }
  
  /** serialize this field as ID reference */
  @JsonIgnore(false)
  @JsonIdentityReference(alwaysAsId = true)
  public Document getDocumentRef() {
    return super.getDocumentRef();
  }
  
  @Override
  @JsonIgnore
  public String getText() {
    return super.getText();
  }
  
  /**
   * Set a score to this result which is used to sort the result list.
   */
  public void setScore(Double score) {
    this.score = score;
  }
  
  /**
   * Get the assigned score for this result, can be NULL.
   */
  public Double getScore() {
    return score;
  }
  
  /**
   * @return a relevence Value, which is assumed to be >0 if the result is relevant, 0 otherwise
   */
  public abstract Integer getRelevance();
  
  /**
   * @return True, if the result is relevant
   */
  public abstract boolean isRelevant();
  
  /**
   * Set to TRUE if high scores mean better ranking, FALSE otherwise.
   */
  protected void setSortDescending(boolean sortDescending) {
    this.sortDescending = sortDescending;
  }
  
  @JsonIgnore
  public boolean isSortDescending() {
    return sortDescending;
  }
  
  public boolean matches(Result other) {
    return this.getDocumentRef().equals(other.getDocumentRef()) &&
      super.matches(other, Match.STRONG);
  }
  
  @Override
  public int compareTo(Span other) {
    if(sortDescending) return resultComparator.compare((Result)other, this);
    else return resultComparator.compare(this, (Result)other);
  }
  
}