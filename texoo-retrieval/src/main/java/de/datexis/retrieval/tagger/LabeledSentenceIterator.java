package de.datexis.retrieval.tagger;

import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.tagger.AbstractMultiDataSetIterator;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A MultiDatasetIterator that returns one Sentence per Example, with Tokens as time steps.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class LabeledSentenceIterator extends AbstractMultiDataSetIterator {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected DocumentFactory df;
  protected Resource source;
  protected LineIterator iterator;
  
  protected WordHelpers.Language lang;
  protected String encoding;
  protected boolean tokenized;
  
  protected static Pattern TAB_SEPARATOR = Pattern.compile("^(.*)\t(.*)$");
  
  public LabeledSentenceIterator(Stage stage, Resource sentencesTSV, String encoding, WordHelpers.Language language, boolean isTokenized, int batchSize, int numExamples, int maxTimeSeriesLength) {
    super(stage, numExamples, maxTimeSeriesLength, batchSize, false);
    this.df = DocumentFactory.getInstance();
    this.lang = language;
    this.encoding = encoding;
    this.tokenized = isTokenized;
    this.source = sentencesTSV;
    if(numExamples < 0) {
      try {
        this.numExamples = Files.lines(sentencesTSV.getPath()).count();
      } catch(IOException e) {
        log.error(e.getMessage());
      }
    }
    reset();
  }
  
  @Override
  public void reset() {
    try {
      if(iterator != null) iterator.close();
      iterator = IOUtils.lineIterator(source.getInputStream(), encoding);
    } catch(IOException e) {
      log.error(e.getMessage());
    }
    super.reset();
  }
  
  protected boolean hasNextSentence() {
    return iterator != null && iterator.hasNext();
  }
  
  @Override
  public boolean hasNext() {
    return hasNextSentence() && !reachedEnd();
  }
  
  public List<String> getLabels() {
    reset();
    List<String> result = new LinkedList<>();
    while(hasNext()) {
      result.add(nextLabeledSentence().getKey());
    }
    reset();
    return result;
  }
  
  public Map.Entry<String, Sentence> nextLabeledSentence() {
    cursor++;
    String text = iterator.next();
    Matcher m = TAB_SEPARATOR.matcher(text);
    if(!m.matches()) {
      log.warn("Could not read line '{}'", text);
      return new AbstractMap.SimpleEntry<>(new String(), new Sentence());
    }
    String label = m.group(1);
    text = m.group(2);
    Sentence s = tokenized ?
      DocumentFactory.createSentenceFromTokenizedString(text) :
      DocumentFactory.createSentenceFromString(text, lang.toString());
    return new AbstractMap.SimpleEntry<>(label, s);
  }
  
  /**
   * Returns the next batch of sentences.
   * @param num - batch size
   * @return List of Sentences and the size of the longest sentence (in Tokens)
   */
  public LabeledSentenceBatch nextSentenceBatch(int num) {
    Map.Entry<String, Sentence> example;
    ArrayList<Sentence> sentences = new ArrayList<>(num);
    ArrayList<String> labels = new ArrayList<>(num);
    int exampleSize = 1; // guarantee to to not return a zero-size dataset
    for(int batchNum=0; batchNum<num; batchNum++) {
      if(hasNext()) example = nextLabeledSentence();
      else example = new AbstractMap.SimpleEntry<>(new String(), new Sentence());
      labels.add(example.getKey());
      sentences.add(example.getValue());
      exampleSize = Math.max(exampleSize, example.getValue().countTokens());
    }
    return new LabeledSentenceBatch(num, sentences, labels, exampleSize, null);
  }
  
  public LabeledSentenceBatch nextSentenceBatch() {
    return nextSentenceBatch(batchSize);
  }
  
  public class LabeledSentenceBatch {
    public List<Sentence> sentences;
    public List<String> labels;
    public MultiDataSet dataset;
    public int size;
    public int maxSentenceLength;
    public LabeledSentenceBatch(int batchSize, List<Sentence> sentences, List<String> labels, int maxSentenceLength, MultiDataSet dataset) {
      this.size = batchSize;
      this.sentences = sentences;
      this.labels = labels;
      this.dataset = dataset;
      this.maxSentenceLength = maxSentenceLength;
    }
  }
  
  @Override
  public MultiDataSet next(int num) {
    LabeledSentenceBatch batch = nextSentenceBatch(num);
    reportProgress(batch.maxSentenceLength);
    return generateDataSet(batch);
  }
  
  public abstract MultiDataSet generateDataSet(LabeledSentenceBatch batch);
  
}
