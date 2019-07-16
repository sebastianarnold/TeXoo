package de.datexis.ner;

import de.datexis.annotator.Annotator;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Token;
import net.amygdalum.stringsearchalgorithms.search.MatchOption;
import net.amygdalum.stringsearchalgorithms.search.StringFinder;
import net.amygdalum.stringsearchalgorithms.search.StringMatch;
import net.amygdalum.stringsearchalgorithms.search.chars.SetBackwardOracleMatching;
import net.amygdalum.stringsearchalgorithms.search.chars.StringSearchAlgorithm;
import net.amygdalum.util.io.CharProvider;
import net.amygdalum.util.io.StringCharProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * An Annotator that creates MentionAnnotations based on a term list for String matching.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MatchingAnnotator extends Annotator {

  protected final static Logger log = LoggerFactory.getLogger(MatchingAnnotator.class);
  
  public static enum MatchingStrategy { CASE_SENSITIVE, LOWERCASE, LEMMA, SKIP_STOPWORDS };

  protected int minimumWordLength = 3; // absolute minimum word length
  protected String type = MentionAnnotation.Type.GENERIC;

  protected Pattern wordLengthMatcher = Pattern.compile("\\b\\w{4,}\\b"); // matches words of length > 3, so that "UPS" will never match "ups"
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
  
  public MatchingAnnotator(MatchingStrategy matchLowercase, Annotation.Source source, String type) {
    this(matchLowercase, source);
    this.type = type;
  }
  
  public MatchingAnnotator(MatchingStrategy matchLowercase, Annotation.Source source, String type, int minWordLength) {
    this(matchLowercase, source, type);
    this.minimumWordLength = minWordLength;
  }
  
  protected Collection<String> convertTerms(Stream<String> terms) {
    switch(matchingStrategy) {
      case LOWERCASE:
        return terms.filter(w -> w.length() >= minimumWordLength).map(w -> convertToLowercase(w)).distinct().collect(Collectors.toList());
      case LEMMA:
        return terms.filter(w -> w.length() >= minimumWordLength).map(w -> removePlurals(convertToLowercase(w))).distinct().collect(Collectors.toList());
      case SKIP_STOPWORDS:
        return terms.filter(w -> w.length() >= minimumWordLength && !wordHelpers.isStopWord(w)).distinct().collect(Collectors.toList());
      default:
        return terms.distinct().collect(Collectors.toList());
    }
  }
  
  public void clearTermsToMatch() {
    this.terms.clear();
    stringSearch = new SetBackwardOracleMatching(this.terms);
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
    if(path.isDirectory()) {
      Files.walk(path.getPath())
        .filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))
        //.filter(p -> p.getFileName().toString().matches(".+"))
        .forEach(p -> {
          try {
            loadTermsToMatch(Resource.fromFile(p.toString()));
          } catch(IOException ex) {
            // IOException is now allowed in Stream
            log.error(ex.toString());
          }
        });
    } else if(path.isFile()) {
      try (BufferedReader br = new BufferedReader(new InputStreamReader(path.getInputStream(), "UTF-8"))) {
        loadTermsToMatch(br.lines());
      }
    } else throw new FileNotFoundException("cannot open path: " + path.toString());
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
    // TODO: use OpenNLP Lemmatizer
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
      StringFinder finder = stringSearch.createFinder(chars, MatchOption.LONGEST_MATCH, MatchOption.NON_OVERLAP);
      for(StringMatch match : finder.findAll()) {
        int begin = (int)match.start();
        int end = (int)match.end();
        final List<Token> list = doc.streamTokensInRange(begin, end, true).collect(Collectors.toList());
        if(spanIsAtTokenBoundaries(list, begin, end, doc)) {
          MentionAnnotation ann = new MentionAnnotation(source, list);
          ann.setType(type);
          doc.addAnnotation(ann);
          // check if there is another overlapping annotation - should not be required with NON_OVERLAP
          /*Collection<MentionAnnotation> existing = doc.getAnnotationsForSpan(source, MentionAnnotation.class, ann);
          if(existing.size() > 1) {
            log.warn("removing overlapping Annotation");
            existing.forEach(a -> doc.removeAnnotation(a));
            ann = Collections.max(existing, Comparator.comparing(MentionAnnotation::getLength));
            doc.addAnnotation(ann);
          }*/
        }
      }
    }
  }

  /**
   * @return True, if given span is exactly at a word boundary.
   */
  private boolean spanIsAtTokenBoundaries(List<Token> list, int begin, int end, Document doc) {
    if(list.isEmpty()) return false;
    else if(list.size() == 1 && list.get(0).getBegin() == begin && list.get(0).getEnd() == end) return true;
    else return list.get(0).getBegin() == begin && list.get(list.size() - 1).getEnd() == end;
  }
  
  
}
