package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import de.datexis.preprocess.DocumentFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A Query is used to query a Dataset to retrieve Results.
 * Results are attached to the Query as Result objects that contain a pointer to the
 * result Document and GOLD and PRED Annotations that point to a Span in this Document.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties({"begin", "length"})
public class Query extends Document {
  
  protected final static Logger log = LoggerFactory.getLogger(Query.class);
  
  /** thread-safe list of results */
  public PriorityBlockingQueue<Result> results;
  
  /**
   * Create a new Document from plain text
   */
  public static Query create(String text) {
    Query result = new Query();
    DocumentFactory.getInstance().addToDocumentFromText(text, result, DocumentFactory.Newlines.KEEP);
    return result;
  }
  
  public Query() {
    results = new PriorityBlockingQueue<>();
  }
  
  /**
   * @return the Annotation that is attached to this Query. Caution: for simplicity of use, we assume there is a single
   * Annotation of this type attached and only this one is returned.
   * @throws NoSuchElementException if there is no Annotation present
   */
  public <A extends Annotation> A getAnnotation(Class<A> type) {
    return streamAnnotations(type).findFirst().get();
  }
  
  /**
   * Add a result to this query
   */
  public <A extends Result> void addResult(A ann) {
    results.add(ann);
  }
  
  public Stream<? extends Result> streamResults() {
    return results.stream().sorted();
  }
  
  public Stream<? extends Result> streamResults(Annotation.Source source) {
    return streamResults()
      .filter(result -> result.getSource().equals(source));
  }
  
  public <A extends Result> Stream<A> streamResults(Annotation.Source source, Class<A> type) {
    return streamResults(source)
      .filter(ann -> type.isAssignableFrom(ann.getClass()))
      .map(ann -> (A) ann);
  }
  
  public List<? extends Result> getResults() {
    return streamResults().collect(Collectors.toList());
  }
  
  public List<? extends Result> getResults(Annotation.Source source) {
    return streamResults(source).collect(Collectors.toList());
  }
  
  public <A extends Result> List<A> getResults(Annotation.Source source, Class<A> type) {
    return streamResults(source, type).collect(Collectors.toList());
  }
  
}
