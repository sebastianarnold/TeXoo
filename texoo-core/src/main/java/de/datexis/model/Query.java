package de.datexis.model;

import de.datexis.preprocess.DocumentFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A Query is used to query a Dataset to retrieve Results.
 * Results are attached to the Query as Result objects that contain a pointer to the
 * result Document and GOLD and PRED Annotations that point to a Span in this Document.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class Query extends Document {
  
  protected final static Logger log = LoggerFactory.getLogger(Query.class);
  
  public List<Result> results;
  
  /**
   * Create a new Document from plain text
   */
  public static Query create(String text) {
    Query result = new Query();
    DocumentFactory.getInstance().addToDocumentFromText(text, result, DocumentFactory.Newlines.KEEP);
    return result;
  }
  
  public Query() {
    results = new ArrayList<>();
  }
  
  /**
   * Add a result to this query
   */
  public void addResult(Document doc, Annotation ann) {
    ann.setDocumentRef(doc);
    results.add(new Result(doc, ann));
  }
  
  public Stream<Result> streamResults() {
    return results.stream();
  }
  
  public Stream<Result> streamResults(Annotation.Source source) {
    return streamResults()
      .filter(result -> result.annotation.getSource().equals(source));
  }
  
  public Collection<Result> getResults() {
    return streamResults().collect(Collectors.toList());
  }
  
  public Collection<Result> getResults(Annotation.Source source) {
    return streamResults(source).collect(Collectors.toList());
  }
  
  /**
   * A Result contains the retrieved Document and an Annotation that points to the
   * Span of the Result (e.g. a Paragraph)
   */
  public class Result {
    
    /** The Document that contains the result */
    private Document document;
    
    /** The Annotation on the Document that contains the result */
    private Annotation annotation;
    
    public Result(Document doc, Annotation ann) {
      this.document = doc;
      this.annotation = ann;
    }
  
    /**
     * @return the Document that this result refers to
     */
    public Document getDocument() {
      return document;
    }
  
    /**
     * @return the Annotation that this result refers to
     */
    public Annotation getAnnotation() {
      return annotation;
    }
    
    public double getRelevance() {
      return annotation.getConfidence();
    }
    
    public boolean matches(Result other) {
      return this.getDocument().equals(other.getDocument()) &&
        this.getAnnotation().matches(other.getAnnotation());
    }
    
  }
  
}
