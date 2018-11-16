package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Token;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Span;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * A character n-gram encoder, mostly used for Letter-Trigrams
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LetterNGramEncoder extends LookupCacheEncoder {
  
  /** the number of chars per gram */
  protected int n;
  
  public LetterNGramEncoder() {
    super("TRI");
    log = LoggerFactory.getLogger(LetterNGramEncoder.class);
  }
  
  public LetterNGramEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(LetterNGramEncoder.class);
  }
  
  public LetterNGramEncoder(int n) {
    super("TRI");
    log = LoggerFactory.getLogger(LetterNGramEncoder.class);
    this.n = n;
  }
  
  @Override
  public String getName() {
    return Integer.toString(n)+"-gram Encoder";
  }

  public int getN() {
    return n;
  }
  
  public LetterNGramEncoder setN(int n) {
    this.n = n;
    return this;
  }

  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }

  @Override
  public INDArray encode(String phrase) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    List<String> ngrams = generateNGrams(phrase);
    int i;
    for(String ngram : ngrams) {
       i = getIndex(ngram);
       // best results were seen with no normalization and 1.0 instead of word frequency
       if(i>=0) vector.put(i, 0, 1.0);//cache.wordFrequency(ngram));
    }
    return vector;//.div(vector.norm1(0));
  }

  @Override
  public boolean isUnknown(String word) {
    List<String> ngrams = generateNGrams(word);
    for(String ngram : ngrams) {
      if(!vocab.containsWord(ngram)) return true;
    }
    return false;
  }

  /**
   * Reduce the input to keep only neccesary characters. Output is lowercase!
   * (printable ISO-8859-1 and Windows-1252 chars)
   * @param input
   * @return 
   */
  public String keepOnlyPrintableChars(String input) {
    // Letters, Numbers, Punctuation
    // http://www.regular-expressions.info/unicode.html#category
    return input.replaceAll("[^\\p{L}\\p{N}\\p{P}\\p{Sm}\\p{Sc}]", "").toLowerCase();
  }
  
  /**
   * Generates a list of n-grams for a token, surrounded by #.
   * Example token = "cat", n = 3: [#ca,cat,at#]
   * @param token The token to split, as String
   * @param n The number of characters to use
   * @return A list of n-grams
   */
  public List<String> generateNGrams(String token, int n) {
    String word = "#" + keepOnlyPrintableChars(token) + "#";
    List<String> arr = new ArrayList<>();
    for (int start = 0; start <= word.length() - n; start++) {
      arr.add(word.substring(start, start + n));
    }
    return arr;
  }
  
  public List<String> generateNGrams(String token) {
    return generateNGrams(token, n);
  }
  
  public List<String> getTrigramsFromProbabilityVector(INDArray vec) {
    ArrayList<String> result = new ArrayList<>();
    for(int i=0; i<vec.length(); i++) {
      if(vec.getDouble(i) > 0.5) result.add(getWord(i));
    }
    return result;
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    trainModel(documents, 1);
  }
  
  public void trainModel(Collection<Document> documents, int minWordFrequency) {
    appendTrainLog("Training " + getName() + " model...");
    setModel(null);
    timer.start();
    List<String> ngrams;
    totalWords = 0;
    for(Document doc : documents) {
      for(Token t : doc.getTokens()) {
        ngrams = generateNGrams(t.getText());
        for(String w : ngrams) {
          totalWords++;
          //System.out.println(x + ": " + cache.containsWord(x));
          if(!vocab.containsWord(w)) {
            vocab.addWord(w);
          } else {
            vocab.incrementWordCounter(w);
          }
        }
      }
    }
    int total = vocab.numWords();
    vocab.truncateVocabulary(minWordFrequency); // TODO: truncate
    vocab.updateHuffmanCodes();
    timer.stop();
    appendTrainLog("trained " + vocab.numWords() + " " + n + "-grams (" +  total + " total)", timer.getLong());
    setModelAvailable(true);
  }
  
}
