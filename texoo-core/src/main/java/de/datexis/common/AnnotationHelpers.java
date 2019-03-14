package de.datexis.common;

import com.google.common.collect.Lists;
import de.datexis.model.*;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class AnnotationHelpers {
  
  /**
   * @return all matching Annotations in a given range
   * @param source Origin of the Annotation
   * @param type return only Annotations of the requested Class
   * @param begin return only Annotations in the given range
   * @param end return only Annotations in the given range
   * @param enclosed TRUE to return only completely enclosed Annotations, FALSE to return all Annotations that intersect
   */
  public static <A extends Annotation> Stream<A> streamAnnotationsInRange(Document doc, Annotation.Source source, Class<A> type, int begin, int end, boolean enclosed, boolean includingSubtypes) {
    if(enclosed) return doc.streamAnnotations(source, type, includingSubtypes)
      .filter(a -> a.getBegin() >= begin && a.getEnd() <= end);
    else return doc.streamAnnotations(source, type, includingSubtypes)
      .filter(a -> (begin <= a.getBegin() && end > a.getBegin()) ||
        (begin >= a.getBegin() && end <= a.getEnd() && begin != end) ||
        (begin < a.getEnd() && end >= a.getEnd()));
  };
  
  /**
   * Returns the annotation that has the largest overlapping range
   */
  public static <A extends Annotation> Optional<A> getAnnotationMaxOverlap(Document doc, Annotation.Source source, Class<A> type, Span s, boolean includingSubtypes) {
    Stream<A> anns = AnnotationHelpers.streamAnnotationsInRange(doc, source, type, s.getBegin(), s.getEnd(), false, includingSubtypes); // all intersecting annotations
    return anns.reduce((first,second) -> // find maximum overlapping range
      WordHelpers.getSpanOverlapLength(s,second) > WordHelpers.getSpanOverlapLength(s,first) ? second : first
    );
  }
  
  public static <A extends Annotation> Optional<A> getAnnotationMaxOverlap(Document doc, Annotation.Source source, Class<A> type, Span s) {
    return getAnnotationMaxOverlap(doc, source, type, s, false);
  }
  
  public static <A extends Annotation> Stream<A> streamAnnotationsForSpan(Document doc, Annotation.Source source, Class<A> type, Span s) {
    return streamAnnotationsForSpan(doc, source, type, s, false);
  }
  
  public static <A extends Annotation> Collection<A> getAnnotationsForSpan(Document doc, Annotation.Source source, Class<A> type, Span s) {
    return streamAnnotationsForSpan(doc, source, type, s, false).sorted().collect(Collectors.toList());
  }
  
  public static <A extends Annotation> Stream<A> streamAnnotationsForSpan(Document doc, Annotation.Source source, Class<A> type, Span s, boolean includingSubtypes) {
    return streamAnnotationsInRange(doc, source, type, s.getBegin(), s.getEnd(), false, includingSubtypes);
  }
  
  public static <A extends Annotation> Collection<A> getAnnotationsForSpan(Document doc, Annotation.Source source, Class<A> type, Span s, boolean includingSubtypes) {
    return streamAnnotationsForSpan(doc, source, type, s, includingSubtypes).sorted().collect(Collectors.toList());
  }
  
  /**
   * Generate pairs for each Span and the GOLD Annotation it is contained in. CAUTION: Annotations need to be non-overlapping.
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
      // we assume that Annotations are continuous and non-overlapping
      if(s.getBegin() >= ann.getEnd() && it.hasNext()) {
        ann = it.next();
      }
      result.add(new AbstractMap.SimpleEntry<S,A>(s, ann));
    }

    return result;
  }
  
  /**
   * Generate pairs for each Span and all GOLD Annotations it is contained in. Will also work for overlapping or sparse Annotations.
   */
  public static <S extends Span, A extends Annotation> List<Map.Entry<S,Collection<A>>> getSpanAnnotationsMultiMap(Document doc, Class<S> spanClass, Class<A> annotationType) {
  
    List<Map.Entry<S,Collection<A>>> result = new ArrayList<>((spanClass == Token.class) ? doc.countTokens() : doc.countSentences());
    
    // iterate over spans and add them to the result
    List<? extends Span> spansToEncode = Collections.EMPTY_LIST;
    if(spanClass == Token.class) spansToEncode = Lists.newArrayList(doc.getTokens());
    else if(spanClass == Sentence.class) spansToEncode = Lists.newArrayList(doc.getSentences());
    else throw new IllegalArgumentException("Span class " + spanClass + " not supported by this method");
    
    for(int t = 0; t < spansToEncode.size(); t++) {
      S span = (S) spansToEncode.get(t);
      Collection<A> anns = AnnotationHelpers.streamAnnotationsForSpan(doc, Annotation.Source.GOLD, annotationType, span, true).collect(Collectors.toList());
      if(!anns.isEmpty()) result.add(new AbstractMap.SimpleEntry<>(span, anns));
    }
    
    return result;
  }
  
  /**
   * Merge all Annotations that overlap or touch.
   * @param anns sorted list of Annotations
   * @return merged Annotations, keeping the attributes of first matched Annotation in group
   */
  public static <A extends Annotation> List<A> mergeAnnotations(List<A> anns) {
    List<A> merged = new ArrayList<>();
    if(anns.size() == 0) return merged;
    A current = anns.get(0);
    for(A ann : anns) {
      // Annotations need to be sorted by begin
      if(ann.intersects(current) || ann.getBegin() == current.getEnd()) {
        current.setBegin(Math.min(ann.getBegin(), current.getBegin()));
        current.setEnd(Math.max(ann.getEnd(), current.getEnd()));
        // TODO: current.setConfidence();
      } else {
        merged.add(current);
        current = ann;
      }
    }
    merged.add(current);
    return merged;
  }
  
}
