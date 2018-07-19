package de.datexis.models.ner;

import de.datexis.annotator.Annotator;
import static net.amygdalum.stringsearchalgorithms.search.MatchOption.LONGEST_MATCH;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Token;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.Comparator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import net.amygdalum.stringsearchalgorithms.search.StringFinder;
import net.amygdalum.stringsearchalgorithms.search.StringMatch;
import net.amygdalum.stringsearchalgorithms.search.chars.*;
import net.amygdalum.util.io.CharProvider;
import net.amygdalum.util.io.StringCharProvider;

/**
 * An Annotator that creates MentionAnnotations based on a term list for String matching.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MatchingAnnotator extends Annotator {

  protected final static Logger log = LoggerFactory.getLogger(MatchingAnnotator.class);
  
  public static enum MatchingStrategy { CASE_SENSITIVE, LOWERCASE, LEMMA, SKIP_STOPWORDS };
  protected final int MIN_WORD_LENGTH = 3;
  
  protected Pattern wordLengthMatcher = Pattern.compile("\\b\\w{4,}\\b"); // matches words of length > 3
  protected Pattern uppercaseMatcher = Pattern.compile("^[A-Z0-9]+$"); // matches uppercase words 
  protected StringSearchAlgorithm stringSearch;
  protected WordHelpers wordHelpers = new WordHelpers(WordHelpers.Language.EN);
  Collection<String> terms = new ArrayList<>();
  
  protected MatchingStrategy matchingStrategy = MatchingStrategy.CASE_SENSITIVE;
  protected Annotation.Source source = Annotation.Source.SILVER;
  
  public MatchingAnnotator() {
    this(MatchingStrategy.CASE_SENSITIVE, Annotation.Source.SILVER);
  }

  public MatchingAnnotator(MatchingStrategy matchingStrategy) {
    this(matchingStrategy, Annotation.Source.SILVER);
  }
  
  public MatchingAnnotator(MatchingStrategy matchLowercase, Annotation.Source source) {
    super();
    this.matchingStrategy = matchLowercase;
    this.source = source;
  }
  
  protected Collection<String> convertTerms(Stream<String> terms) {
    switch(matchingStrategy) {
      case LOWERCASE:
        return terms.filter(w -> w.length() >= MIN_WORD_LENGTH).map(w -> convertToLowercase(w)).distinct().collect(Collectors.toList());
      case LEMMA:
        return terms.filter(w -> w.length() >= MIN_WORD_LENGTH).map(w -> removePlurals(convertToLowercase(w))).distinct().collect(Collectors.toList());
      case SKIP_STOPWORDS:
        return terms.filter(w -> w.length() >= MIN_WORD_LENGTH && !wordHelpers.isStopWord(w)).distinct().collect(Collectors.toList());
      default:
        return terms.distinct().collect(Collectors.toList());
    }
  }
  
  public void loadTermsToMatch(Collection<String> terms) {
    loadTermsToMatch(terms.stream());
  }
  
  public void loadTermsToMatch(Stream<String> terms) {
    this.terms.addAll(convertTerms(terms));
    log.info("Rebuildung dictionary with {} distinct terms", this.terms.size());
    // AhoCorasick - fast for small matches, correct but memory-intensive
    // WuManber - correct matches, but slow
    // SetBackwardOracleMatching - suoer fast, but invalid LONGEST_MATCH
    stringSearch = new SetBackwardOracleMatching(this.terms);
  }
  
  public void loadTermsToMatch(Resource path) throws IOException {
    try (BufferedReader br = new BufferedReader(new InputStreamReader(path.getInputStream(), "UTF-8"))) {
      loadTermsToMatch(br.lines());
    }
  }
  
  public void deleteTermsToMatch(Collection<String> terms) {
    deleteTermsToMatch(terms.stream());
  }
  
  public void deleteTermsToMatch(Stream<String> terms) {
    this.terms.removeAll(convertTerms(terms));
    log.info("Rebuildung dictionary with {} distinct terms", this.terms.size());
    stringSearch = new SetBackwardOracleMatching(this.terms);
  }
  
  public void deleteTermsToMatch(Resource path) throws IOException {
    try (BufferedReader br = new BufferedReader(new InputStreamReader(path.getInputStream(), "UTF-8"))) {
      deleteTermsToMatch(br.lines());
    }
  }
  
  public int countTerms() {
    return terms.size();
  }
  
  /**
   * @return text with all words >3 chars converted to lowercase.
   */
  protected String convertToLowercase(String text) {
    Matcher m = wordLengthMatcher.matcher(text);
    StringBuffer sb = new StringBuffer();
     while(m.find()) {
      String match = m.group();
      Matcher u = uppercaseMatcher.matcher(match);
      if(u.matches()) { // all uppercase
        if(match.length() >= 8) m.appendReplacement(sb, m.group().toLowerCase());
      } else {
        m.appendReplacement(sb, m.group().toLowerCase());
      }
    }
    m.appendTail(sb);
    return sb.toString();
  }
  
  protected String removePlurals(String text) {
    throw new UnsupportedOperationException("Lemma matching is not yet implemented.");
  }
  
  @Override
  public void annotate(Collection<Document> docs) {
    annotate(docs, source);
  }

  /**
   * Annotates a Dataset using the pre-trained list.
   * @param docs - the Documents to annotate
   * @param source - the type of annotations to create, e.g. SILVER
   * @param fuzzyness - set to 0 for exact matching
   */
  public void annotate(Iterable<Document> docs, Annotation.Source source) {
    for(Document doc : docs) {
      // see http://stringsearchalgorithms.amygdalum.net/
      String text = doc.getText();
      if(matchingStrategy.equals(MatchingStrategy.LOWERCASE)) text = convertToLowercase(doc.getText());
      CharProvider chars = new StringCharProvider(text, 0);
      if(stringSearch == null) {
        log.warn("MatchingAnnotator called without terms loaded");
        return;
      }
      StringFinder finder = stringSearch.createFinder(chars, LONGEST_MATCH);
      for(StringMatch match : finder.findAll()) {
        if(spanIsAtTokenBoundaries((int)match.start(), (int)match.end(), doc)) {
          MentionAnnotation ann = new MentionAnnotation(source, match.text(), (int)match.start(), (int)match.end());
          // check if there is another overlapping annotation
          doc.addAnnotation(ann);
          Collection<MentionAnnotation> existing = doc.getAnnotationsForSpan(source, MentionAnnotation.class, ann);
          if(existing.size() > 1) {
            existing.forEach(a -> doc.removeAnnotation(a));
            ann = Collections.max(existing, Comparator.comparing(MentionAnnotation::getLength));
            doc.addAnnotation(ann);
          }
        }
      }
    }
  }

  /**
   * @return True, if given span is exactly at a word boundary.
   */
  private boolean spanIsAtTokenBoundaries(int begin, int end, Document doc) {
    final List<Token> list = doc.streamTokensInRange(begin, end, true).collect(Collectors.toList());
    if(list.isEmpty()) return false;
    else if(list.size() == 1 && list.get(0).getBegin() == begin && list.get(0).getEnd() == end) return true;
    else return list.get(0).getBegin() == begin && list.get(list.size() - 1).getEnd() == end;
  }
  
}
