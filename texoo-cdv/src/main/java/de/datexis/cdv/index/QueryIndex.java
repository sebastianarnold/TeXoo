package de.datexis.cdv.index;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.IEncoder;
import de.datexis.model.Query;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.retrieval.index.InMemoryIndex;
import de.datexis.retrieval.tagger.LSTMSentenceTaggerIterator;
import de.datexis.retrieval.tagger.LabeledSentenceIterator;
import de.datexis.tagger.AbstractMultiDataSetIterator;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Embedding target index implementation for CDV
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class QueryIndex extends InMemoryIndex {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  /** called from JSON deserialization */
  protected QueryIndex() {}
  
  public QueryIndex(TokenPreProcess keyPreprocessor, IEncoder encoder) {
    super(keyPreprocessor, encoder);
  }
  
  public abstract void encodeFromQueries(Collection<Query> queries);
  
  public void clear() {
    this.keyVocabulary = new AbstractCache.Builder()
      .hugeModelExpected(false)
      .minElementFrequency(0)
      .build();
    this.lookupVectors = new InMemoryLookupTable<>(keyVocabulary, (int)getEmbeddingVectorSize(), true, 0.01, Nd4j.getRandom(), 0, true);
  }
  
  public void buildKeyIndex(Resource sentencesTSV) {
    LSTMSentenceTaggerIterator it = new LSTMSentenceTaggerIterator(AbstractMultiDataSetIterator.Stage.ENCODE, null, null, sentencesTSV, "utf-8", WordHelpers.Language.EN, true, 1);
    buildKeyIndex(it.getLabels());
  }
  
  /**
   * Train a model by averaging over all sentences per label in a given TSV file <label>\tab<sentence>
   */
  public void encodeIndexFromSentences(Resource sentencesTSV) {
    encodeIndexFromSentences(sentencesTSV, Collections.emptySet(), true);
  }
  
  public void encodeIndexFromSentences(Resource sentencesTSV, Set<String> stopWords, boolean isTokenized) {
    LabeledSentenceIterator it = new LSTMSentenceTaggerIterator(AbstractMultiDataSetIterator.Stage.ENCODE, encoder, null, sentencesTSV, "utf-8", WordHelpers.Language.EN, stopWords, isTokenized, 1);
    log.info("Reading {} examples...", it.getNumExamples());
    Multimap<String, Sentence> examples = ArrayListMultimap.create();
    String key;
    while(it.hasNext()) {
      Map.Entry<String, Sentence> example = it.nextLabeledSentence();
      key = keyPreprocessor.preProcess(example.getKey());
      examples.put(key, example.getValue()); // -heading
    }
    buildKeyIndex(examples.keys(), false);
    encodeAndBuildVectorIndex(examples, false);
    setModelAvailable(true);
  }
  
  /**
   * Train a model by averaging over all labels in a given TSV file <label>\tab<sentence>
   */
  public void encodeIndexFromLabels(Resource sentencesTSV) {
    LabeledSentenceIterator it = new LSTMSentenceTaggerIterator(AbstractMultiDataSetIterator.Stage.ENCODE, encoder, null, sentencesTSV, "utf-8", WordHelpers.Language.EN, true, 64);
    List<String> labels = it.getLabels();
    Multimap<String, Sentence> examples = ArrayListMultimap.create();
    for(String label : labels) {
      String key = keyPreprocessor.preProcess(label);
      Sentence span = DocumentFactory.createSentenceFromTokenizedString(label);
      if(!examples.containsKey(key))
        examples.put(key, span); // just a simple mapping from heading to heading!
    }
    buildKeyIndex(examples.keys(), false);
    encodeAndBuildVectorIndex(examples, false);
    setModelAvailable(true);
  }
  
  /**
   * Sample index entries based on the probability of a given key,
   * so that rare entries are assigned 1 and the most frequent ones around 0.1
   * @return sampling factor
   */
  public double weightFactor(String key) {
    double alpha = 0.03;
    double p = probability(key);
    //return alpha / (alpha + p);
    //return Math.min(1, (Math.sqrt(p / alpha) + 1) * (alpha / p));
    return Math.min(1, alpha / Math.sqrt(p)); // worked well with alpha = 0.03
  }
  
}
