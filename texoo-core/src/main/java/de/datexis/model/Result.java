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
public abstract class Result extends Annotation {
  
  protected static Comparator<Result> resultComparator =
    Comparator.comparing(Result::getRank, Comparator.nullsLast(Integer::compare)) // lowest rank first
      .thenComparing(Result::getScore, Comparator.nullsLast(Double::compare).reversed()) // highest score first
      .thenComparing(Result::getRelevance, Comparator.nullsLast(Integer::compare).reversed()); // highest relevance first
  
  /* assigned ID for this result*/
  protected String id = null;
  
  /** score of this Result, highest ist best, NULL to ignore */
  protected Double score = null;
  
  /** calculated rank of this result, lowest is best, NULL if unknown */
  protected Integer rank = null;
  
  /** reference to an existing Annotation in a Document */
  protected Annotation annotationRef = null;
  
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
  
  @JsonIgnore
  public Annotation getAnnotationRef() {
    return annotationRef;
  }
  
  public void setAnnotationRef(Annotation ann) {
    this.annotationRef = ann;
  }
  
  @Override
  @JsonIgnore
  public double getConfidence() {
    return super.getConfidence();
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
   * Set the ranking position 1..N
   */
  public void setRank(Integer rank) {
    this.rank = rank;
  }
  
  /**
   * @return the ranking position 1..N or NULL if not ranked yet
   */
  public Integer getRank() {
    return rank;
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
   * @return TRUE, iff the referred Document and offsets match exactly
   */
  public boolean matches(Result other) {
    return this.getDocumentRef().equals(other.getDocumentRef()) &&
      super.matches(other, Match.STRONG);
  }
  
  @Override
  public int compareTo(Span other) {
    return resultComparator.compare(this, (Result)other);
  }
  
}