package de.datexis.cdv.retrieval;

import com.google.common.collect.Lists;
import de.datexis.model.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class QueryHelpers {
  
  protected final static Logger log = LoggerFactory.getLogger(QueryHelpers.class);
  
  /**
   * Generate pairs for each Span and Annotation type.
   */
  public static <S extends Span, A extends Result> List<Map.Entry<S, Optional<A>>> getSpanResultsMap(Document doc, de.datexis.model.Query query, Annotation.Source source, Class<S> spanClass, Class<A> annotationType) {
    
    List<Map.Entry<S,Optional<A>>> result = new ArrayList<>();
    
    // gather all Results of requested Type
    List<A> anns = query
      .streamResults(source, annotationType)
      .filter(r -> r.getDocumentRef() == doc) // reference should be faster than equals()
      .sorted()
      .collect(Collectors.toList());
    
    Iterator<A> it = anns.iterator();
    if(!it.hasNext()) return result; // no annotations
    
    // iterate over spans and add them to the result
    List<? extends Span> spansToEncode = Collections.EMPTY_LIST;
    if(spanClass == Token.class) spansToEncode = Lists.newArrayList(doc.getTokens());
    else if(spanClass == Sentence.class) spansToEncode = Lists.newArrayList(doc.getSentences());
    else throw new IllegalArgumentException("Span class " + spanClass + " not supported by this method");
    
    A ann = it.next();
    for(int t = 0; t < spansToEncode.size(); t++) {
      S s = (S) spansToEncode.get(t);
      if(s.getBegin() >= ann.getEnd() && it.hasNext()) {
        ann = it.next();
      }
      if(ann.contains(s)) {
        result.add(new AbstractMap.SimpleEntry<S,Optional<A>>(s, Optional.of(ann)));
      } else {
        result.add(new AbstractMap.SimpleEntry<S,Optional<A>>(s, Optional.empty()));
      }
    }
    
    return result;
  }
  
  /**
   * @return a sorted list of query results of requested type on the given document
   */
  public static <A extends Result> List<A> getResultsForDoc(Document doc, de.datexis.model.Query query, Annotation.Source source, Class<A> annotationType) {
    // gather all Results of requested Type
    return query
      .streamResults(source, annotationType)
      .filter(r -> r.getDocumentRef() == doc) // reference should be faster than equals()
      .sorted()
      .collect(Collectors.toList());
  }
  
  
}
