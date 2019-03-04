package de.datexis.common;

import com.google.common.collect.Lists;
import de.datexis.model.*;

import java.util.*;
import java.util.stream.Collectors;

public class AnnotationHelpers {

  /**
   * Generate pairs for each Span and Annotation type.
   */
  public static <S extends Span, A extends Annotation> List<Map.Entry<S,A>> getSpanAnnotationsMap(Document doc, Class<S> spanClass, Class<A> annotationType) {

    List<Map.Entry<S,A>> result = new ArrayList<>();

    // gather all Annotations of requested Type
    List<A> anns = doc
            .streamAnnotations(Annotation.Source.GOLD, annotationType)
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
      result.add(new AbstractMap.SimpleEntry<S,A>(s, ann));
    }

    return result;
  }

}
