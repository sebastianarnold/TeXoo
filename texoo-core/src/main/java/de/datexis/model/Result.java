package de.datexis.model;

import java.util.Comparator;

/**
 * A Result contains the retrieved Document and an Annotation that points to the
 * Span of the Result (e.g. a Paragraph)
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class Result/*<S extends Comparable> */extends Annotation {
  
//  protected static Comparator<Double> nullsafeDoubleComparator =
//  Comparator.nullsFirst(Double::compare);
  
  protected static Comparator<Result> resultComparator =
    Comparator.comparing(Result::getScore, Comparator.nullsFirst(Double::compare));
  
  protected /*S*/ Double score = null;
  protected boolean sortDescending = true;
  
  public Result(Source source) {
    super(source, "");
  }
  
  public Result(Source source, Document doc, int begin, int end) {
    super(source, "");
    this.setDocumentRef(doc);
    this.setBegin(begin);
    this.setEnd(end);
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
   * Set to TRUE if high scores mean better ranking, FALSE otherwise.
   */
  protected void setSortDescending(boolean sortDescending) {
    this.sortDescending = sortDescending;
  }
  
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