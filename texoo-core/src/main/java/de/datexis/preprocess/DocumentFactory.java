package de.datexis.preprocess;

import com.google.common.base.Optional;
import com.optimaize.langdetect.LanguageDetector;
import com.optimaize.langdetect.LanguageDetectorBuilder;
import com.optimaize.langdetect.i18n.LdLocale;
import com.optimaize.langdetect.ngram.NgramExtractors;
import com.optimaize.langdetect.profiles.LanguageProfile;
import com.optimaize.langdetect.profiles.LanguageProfileReader;
import com.optimaize.langdetect.text.CommonTextObjectFactories;
import com.optimaize.langdetect.text.TextObject;
import com.optimaize.langdetect.text.TextObjectFactory;
import de.datexis.common.WordHelpers;
import java.util.ArrayList;
import java.util.List;
import static de.datexis.common.WordHelpers.skipSpaceAfter;
import static de.datexis.common.WordHelpers.skipSpaceBefore;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.PTBTokenizer.PTBTokenizerFactory;
import edu.stanford.nlp.process.WordToSentenceProcessor;
import edu.stanford.nlp.process.WordTokenFactory;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Generics;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.Iterator;
import java.util.Properties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Creates a fully tokenized Document from raw text or Stanford Tokens.
 * @author sarnold, fgrimme
 */
public class DocumentFactory {

  protected final static Logger log = LoggerFactory.getLogger(DocumentFactory.class);
  public static final int AVERAGE_ENGLISH_SENTENCE_LENGTH = 14;

  protected static DocumentFactory instance = new DocumentFactory();
  
  public static DocumentFactory getInstance() {
    return instance;
  }
  
  public static enum Newlines { KEEP, KEEP_DOUBLE, DISCARD };
  
	private final StanfordCoreNLP pipeline;
  WordToSentenceProcessor<Token> tts;
  WordToSentenceProcessor<Word> wts;
  WordTokenFactory wtf;
  PTBTokenizerFactory<Word> ptf;
  TextObjectFactory textObjectFactory;
  LanguageDetector languageDetector;
  
  /**
   * Create a new DocumentFactory instance. Use this only if you need multiple instances!
   * Otherwise, getInstance() will return a singleton object that you can use.
   */
  public DocumentFactory() {
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize, ssplit");
    pipeline = new StanfordCoreNLP(props);
    tts = new WordToSentenceProcessor<>(WordToSentenceProcessor.NewlineIsSentenceBreak.ALWAYS);
    //wts = new WordToSentenceProcessor<>(WordToSentenceProcessor.NewlineIsSentenceBreak.ALWAYS);
    wts = new WordToSentenceProcessor<>(WordToSentenceProcessor.DEFAULT_BOUNDARY_REGEX + "|\\*NL\\*", WordToSentenceProcessor.DEFAULT_BOUNDARY_FOLLOWERS_REGEX + "|\\*NL\\*",
            Generics.newHashSet(), null, null, WordToSentenceProcessor.NewlineIsSentenceBreak.ALWAYS, null, null, false, false);
    wtf = new WordTokenFactory();
    // see http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/process/PTBTokenizer.html
    ptf = PTBTokenizerFactory.newWordTokenizerFactory("tokenizeNLs=true,ptb3Escaping=false,americanize=false,"
              + "normalizeParentheses=false,normalizeOtherBrackets=false,asciiQuotes=true,latexQuotes=false,"
              + "escapeForwardSlashAsterisk=false,untokenizable=noneDelete");
    
    try {
      //load all languages:
      List<LanguageProfile> languageProfiles = new LanguageProfileReader().readAllBuiltIn();
      //build language detector:
      languageDetector = LanguageDetectorBuilder.create(NgramExtractors.standard())
              .withProfiles(languageProfiles)
              .build();
      //create a text object factory
      textObjectFactory = CommonTextObjectFactories.forDetectingOnLargeText();
    } catch (IOException ex) {
      log.error("Could not load language profiles");
    }
  }
  
  /**
   * Creates a Document with Sentences and Tokens from a String.
   * Uses Stanford CoreNLP PTBTokenizerFactory and removes Tabs, Newlines and trailing whitespace.
   * Use fromText(text, true) to keep the original String in memory.
   * @param text
   * @return 
   */
  public static Document fromText(String text) {
		return instance.createFromText(text);
	}
  
  public static Document fromText(String text, Newlines newlines) {
		return instance.createFromText(text, newlines);
	}
  
  /**
   * Creates a Document from existing Tokens, processing Span positions and Sentence splitting.
   */
  public static Document fromTokens(List<Token> tokens) {
		return instance.createFromTokens(tokens);
	}
  
  /**
   * Create Tokens from raw text, without sentence splitting.
   */
  public static List<Token> createTokensFromText(String text) {
		return instance.tokenizeFast(text);
	}
  
  /**
   * Create Tokens from tokenized text, without sentence splitting.
   */
  public static List<Token> createTokensFromTokenizedText(String text) {
    return instance.createTokensFromTokenizedText(text, 0);
  }
    
  /**
   * Creates a Document with Sentences and Tokens from a String.
   * Uses Stanford CoreNLP PTBTokenizerFactory.
   * @param text
   * @param keepOrig - if TRUE, a copy of the string is saved to keep newlines and tabs.
   * @return 
   */
  public synchronized Document createFromText(String text) {
    Document doc = new Document();
    addToDocumentFromText(text, doc, Newlines.DISCARD);
    return doc;
  }
  
  public synchronized Document createFromText(String text, Newlines newlines) {
    Document doc = new Document();
    addToDocumentFromText(text, doc, newlines);
    return doc;
  }
  
  public synchronized void addToDocumentFromText(String text, Document doc, Newlines newlines) {
    doc.setLanguage(detectLanguage(text));
    int offset = doc.getEnd();
    if(offset > 0) offset++;
    try(Reader r = new StringReader(text)) {
      List<Word> words = ptf.getTokenizer(r).tokenize();
      words = fixTokenization(words);
      int countNewlines = 0;
      for(List<Word> sentence : wts.process(words)) {
        Sentence s = new Sentence();
        for(Word w : sentence) {
          if(w.word().equals("*NL*")) { // newline
            countNewlines++;
            if(newlines == Newlines.KEEP) { // newline is a paragraph
              s.addToken(new Token("\n", w.beginPosition() + offset, w.endPosition() + offset));
            } else if(newlines == Newlines.KEEP_DOUBLE && countNewlines == 2) { // two newlines are a new paragraph, skip next though
              s.addToken(new Token("\n", w.beginPosition() + offset, w.endPosition() + offset));
            } else if(newlines == Newlines.DISCARD) { // skip newlines, but keep one whitespace
              if(countNewlines > 1) offset--;
            } else {
              offset--;
            }
          /*} else if(w.word().equals("*EOS*")) { // end of sentence
            offset--;
            // nop*/
          } else if(w.word().trim().isEmpty()) { 
            // nop
          } else {
            s.addToken(new Token(w.word(), w.beginPosition() + offset, w.endPosition() + offset));
            countNewlines = 0;
          }
        }
        if(!s.isEmpty()) doc.addSentence(s, false);
      }
    } catch (IOException ex) {
      log.error(ex.toString());
    } 
  }
  
  public synchronized List<Token> tokenizeFast(String text) {
    ArrayList<Token> result = new ArrayList<>();
    try(Reader r = new StringReader(text)) {
      List<Word> words = ptf.getTokenizer(r).tokenize();
      words = fixTokenization(words);
      for(Word w : words) {
        if(!w.word().trim().isEmpty()) result.add(new Token(w.word(), w.beginPosition(), w.endPosition()));
      }
    } catch (IOException ex) {
      log.error(ex.toString());
    }
    return result;
  }
  
  public static String getLanguage(String text) {
    return instance.detectLanguage(text);
  }
  
  public synchronized String detectLanguage(String text) {
    try {
      TextObject textObject = textObjectFactory.forText(text);
      Optional<LdLocale> locale = languageDetector.detect(textObject);
      if(locale.isPresent()) return locale.get().getLanguage();
    } catch(Exception e) {}
    return "";
  }
  
  /**
   * Fixes Tokens that PTBTokenizer fails on. E.g. "20. Januar", "Kaiser XIV."
   * We may skip some sentence boundaries with that, but prefer this over too many false splits.
   * @param words
   * @return 
   */
  private List<Word> fixTokenization(Iterable<Word> words) {
    List<Word> result = new ArrayList<>();
    Iterator<Word> it = words.iterator();
    String split = wts.DEFAULT_BOUNDARY_REGEX;
    String nosplit = "^(\\d{1,3}|[a-zäüö]|[IVXLCDM]+|ggf|evtl|bzw|engl|dpt|griech|lat|allg|bspw|geb)$"; // German abbreviations are not in CoreNLP
    Word last = Word.EMPTY;
    Word word;
    while(it.hasNext()) {
      word = it.next();
      // check for Sentence boundary
      if(word.word().matches(split)) { // word is a sentence boundary, e.g. "."
        if(last.word().matches(nosplit)) { // last word is an abbreviation
          if(last.endPosition() == word.beginPosition()) { // double check if words are not split by whitespace
            last.setWord(last.word() + word.word());
            last.setEndPosition(word.endPosition()); // join words
            word = Word.EMPTY;
          }
        } /*else if(last.word().endsWith(".") && !last.word().endsWith("..")) { // this is a duplicate sentence end added by CoreNLP
          word = new Word("*EOS*", word.beginPosition(), word.endPosition());
        }*/
      }
      if(!last.equals(Word.EMPTY)) result.add(last);
      last = word;
    }
    if(!last.equals(Word.EMPTY)) result.add(last);
    return result;
  }
  
public Document createFromTokens(List<Token> tokens) {
    Document doc = new Document();
    createSentencesFromTokens(tokens).forEach(sentence -> {
      doc.addSentence(sentence, false);
    });
    doc.setLanguage(detectLanguage(doc.getText()));
    return doc;
  }

  public static Sentence createSentenceFromTokens(List<Token> sentence) {
    return instance.createSentenceFromTokens(sentence, "", 0);
  }
  
  public List<Sentence> createSentencesFromTokens(List<Token> tokens) {
    List<Sentence> sentences = new ArrayList<>(tokens.size() / AVERAGE_ENGLISH_SENTENCE_LENGTH);
    int lastCursorPos = 0;
    String lastTokenString = "";
    for(List<Token> sentenceTokens : instance.tts.process(tokens)) {
      Sentence sentence = createSentenceFromTokens(sentenceTokens, lastTokenString, lastCursorPos);
      lastCursorPos = sentence.getEnd();
      int lastTokenIndex = sentence.getTokens().size() - 1;
      lastTokenString = sentence.getToken(lastTokenIndex).getText();
      sentences.add(sentence);
    }
    return sentences;
  }
  
  private Sentence createSentenceFromTokens(List<Token> sentence, String last, Integer cursor) {
    int length;
    Sentence s = new Sentence();
    s.setBegin(cursor);
    for(Token t : sentence) {
      if(!skipSpaceAfter.contains(last) && !skipSpaceBefore.contains(t.getText())) cursor++;
      length = t.getText().length();
      t.setBegin(cursor);
      t.setLength(length);
      cursor += length;
      last = t.getText();
      s.addToken(t);
    }
    s.setEnd(cursor);
    return s;
  }

  /**
   * Creates a list of Tokens from raw text (ignores sentences)
   */
  public List<Token> createTokensFromText(String text, int offset) {
    List<Token> tokens = new ArrayList<>();
    edu.stanford.nlp.pipeline.Annotation document = new edu.stanford.nlp.pipeline.Annotation(text);
    pipeline.annotate(document);
    for(CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
      for(CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        int length = token.get(CoreAnnotations.CharacterOffsetEndAnnotation.class) - token.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
        Token t = new Token(token.originalText(), offset, offset + length);
        offset += length;
        tokens.add(t);
      }
    }
    return tokens;
  }
  
  /**
   * Creates a list of Tokens from tokenized text, keeping the original tokenization.
   */
  public List<Token> createTokensFromTokenizedText(String text, int offset) {
    List<Token> tokens = new ArrayList<>();
    String last = "";
    for(String token : WordHelpers.splitSpaces(text)) {
      int length = token.length();
      Token t = new Token(token, offset, offset + length);
      offset += length;
      if(!skipSpaceAfter.contains(last) && !skipSpaceBefore.contains(token)) offset++;
      tokens.add(t);
      last = token;
    }
    return tokens;
  }
  
  /**
   * Recreates the document with automatic tokenization. Offsets are kept.
   */
  public void retokenize(Document doc) {
    doc.setText(doc.getText());
  }
  
  /**
   * tokenizes the document into a list of words and punctuations
   * @param text The document as String
   * @return List of words
   */
  public List<CoreLabel> createCoreLabels(String text) {
    edu.stanford.nlp.pipeline.Annotation document = new edu.stanford.nlp.pipeline.Annotation(text);
    pipeline.annotate(document);
    List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
    List<CoreLabel> tokens = new ArrayList<>();
    for(CoreMap sentence : sentences) {
      for(CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        tokens.add(token);
      }
    }
    return tokens;
  }
    
}
