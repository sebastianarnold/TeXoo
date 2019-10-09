package de.datexis.retrieval.tagger;

import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.EncodingHelpers;
import de.datexis.encoder.IEncoder;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * Iterates through a Dataset with Document-Level Batches of Sentences
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LSTMSentenceTaggerIterator extends LabeledSentenceIterator {

  protected IEncoder inputEncoder, targetEncoder;
  
  protected Set<String> stopWords = Collections.emptySet();
  
  public LSTMSentenceTaggerIterator(Stage stage, IEncoder inputEncoder, IEncoder targetEncoder, Resource sentencesTSV, String encoding, WordHelpers.Language language, boolean isTokenized, int batchSize) {
    this(stage, inputEncoder, targetEncoder, sentencesTSV, encoding, language, Collections.emptySet(), isTokenized, batchSize, -1, -1);
  }
  
  public LSTMSentenceTaggerIterator(Stage stage, IEncoder inputEncoder, IEncoder targetEncoder, Resource sentencesTSV, String encoding, WordHelpers.Language language, Set<String> stopWords, boolean isTokenized, int batchSize) {
    this(stage, inputEncoder, targetEncoder, sentencesTSV, encoding, language, stopWords, isTokenized, batchSize, -1, -1);
  }
  
  public LSTMSentenceTaggerIterator(Stage stage, IEncoder inputEncoder, IEncoder targetEncoder, Resource sentencesTSV, String encoding, WordHelpers.Language language, Set<String> stopWords, boolean isTokenized, int batchSize, int numExamples, int maxTimeSeriesLength) {
    super(stage, sentencesTSV, encoding, language, isTokenized, batchSize, numExamples, maxTimeSeriesLength);
    this.inputEncoder = inputEncoder;
    this.targetEncoder = targetEncoder;
    this.stopWords = stopWords;
  }
  
  /** no file reading, used for encoding only */
  public LSTMSentenceTaggerIterator(Stage stage, IEncoder inputEncoder, IEncoder targetEncoder, Set<String> stopWords, int batchSize, int numExamples, int maxTimeSeriesLength) {
    super(stage, batchSize, numExamples, maxTimeSeriesLength);
    this.inputEncoder = inputEncoder;
    this.targetEncoder = targetEncoder;
    this.stopWords = stopWords;
  }
  
  public LabeledSentenceBatch applyStopWordFilter(LabeledSentenceBatch batch) {
    if(!stopWords.isEmpty()) {
      List<String> labels = batch.labels;
      List<Sentence> examples = batch.sentences.stream()
        .map(s -> applyStopWordFilter(s))
        .collect(Collectors.toList());
      int maxSentenceLength = 1;
      for(int i = examples.size() - 1; i >= 0; i--) {
        Sentence s = examples.get(i);
        if(s.countTokens() > 0) {
          maxSentenceLength = Math.max(maxSentenceLength, s.countTokens());
        } else {
          examples.remove(i);
          labels.remove(i);
        }
      }
      batch.sentences = examples;
      batch.labels = labels;
      batch.size = examples.size();
      batch.maxSentenceLength = maxSentenceLength;
    }
    return batch;
  }
  
  public Sentence applyStopWordFilter(Sentence s) {
    return new Sentence(s.streamTokens()
          .filter(t -> !stopWords.contains(t.getText().toLowerCase().trim()))
          .collect(Collectors.toList()));
  }
  
  public LabeledSentenceBatch nextSentenceBatch(int num) {
    return applyStopWordFilter(super.nextSentenceBatch(num));
  }
  
  public Map.Entry<String, Sentence> nextLabeledSentence() {
    Map.Entry<String, Sentence> next = super.nextLabeledSentence();
    if(!stopWords.isEmpty()) next.setValue(applyStopWordFilter(next.getValue()));
    return next;
  }
  
  @Override
  public MultiDataSet generateDataSet(LabeledSentenceBatch batch) {
    
    // input encodings
    INDArray inputMask = createMask(batch.sentences, batch.maxSentenceLength, Token.class); // activate all Tokens
    INDArray labelsMask = createLabelsMask(batch.sentences, Token.class); // activate Sentences in batch

    // return all encodings on Token level
    INDArray input = EncodingHelpers.encodeTimeStepMatrix(batch.sentences, inputEncoder, batch.maxSentenceLength, Token.class);

    // target encodings
    INDArray targets;
    if(stage.equals(Stage.TRAIN) || stage.equals(Stage.TEST)) targets = encodeTarget(batch.sentences, batch.labels);
    else targets = Nd4j.zeros(DataType.FLOAT, batch.size, targetEncoder.getEmbeddingVectorSize());
  
    return new org.nd4j.linalg.dataset.MultiDataSet(
      new INDArray[]{input},
      new INDArray[]{targets},
      new INDArray[]{inputMask},
      new INDArray[]{labelsMask}
    );
  }
  
  /**
   * @return a feature mask which activates every Token in a Sentence
   */
  public static INDArray createMask(List<Sentence> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    INDArray mask = Nd4j.zeros(DataType.FLOAT, input.size(), maxTimeSteps);
    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      int spanCount = 0;
      if(timeStepClass == Token.class) spanCount = input.get(batchIndex).countTokens();
      for(int t = 0; t < spanCount && t < maxTimeSteps; t++) {
        mask.putScalar(new int[] {batchIndex, t}, 1.);
      }
    }
    return mask;
  }
  
  public static INDArray createLabelsMask(List<Sentence> input, Class<? extends Span> timeStepClass) {
    INDArray mask = Nd4j.zeros(DataType.FLOAT, input.size(), 1);
    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      int spanCount = 0;
      if(timeStepClass == Token.class) spanCount = input.get(batchIndex).countTokens();
      if(spanCount > 0) mask.putScalar(new int[] {batchIndex, 0}, 1.);
    }
    return mask;
  }
  
  /**
   * @return a label mask which activates only the first Token of every Sentence
   */
  public INDArray createBwdMask(List<Sentence> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    INDArray mask = Nd4j.zeros(DataType.FLOAT, input.size(), maxTimeSteps);
    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      int spanCount = 0;
      if(timeStepClass == Token.class) spanCount = Math.min(input.get(batchIndex).countTokens(), maxTimeSteps);
      if(spanCount > 0 && maxTimeSteps > 0) mask.putScalar(new int[] {batchIndex, 0}, 1.);
    }
    return mask;
  }
  
  /**
   * @return a label mask which activates only the last Token of every Sentence
   */
  public INDArray createFwdMask(List<Sentence> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    INDArray mask = Nd4j.zeros(DataType.FLOAT, input.size(), maxTimeSteps);
    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      int spanCount = 0;
      if(timeStepClass == Token.class) spanCount = Math.min(input.get(batchIndex).countTokens(), maxTimeSteps);
      if(spanCount > 0 && maxTimeSteps > 0) mask.putScalar(new int[] {batchIndex, spanCount - 1}, 1.);
    }
    return mask;
  }

  public INDArray encodeRNNTarget(List<Sentence> input, List<String> labels, int maxTimeSteps, Class<? extends Span> timeStepClass) {

    INDArray encoding = Nd4j.zeros(DataType.FLOAT, labels.size(), targetEncoder.getEmbeddingVectorSize(), maxTimeSteps);
    String label;
    Sentence example;

    for(int batchIndex = 0; batchIndex < labels.size(); batchIndex++) {
      label = labels.get(batchIndex);
      example = input.get(batchIndex);
      int t = 0;
      INDArray vec = targetEncoder.encode(label);
      //log.debug("target: {} -> {} ({})", label, vec.transpose().toString(), vec.sumNumber().toString());
      for(Token token : example.getTokens()) {
        if(t >= maxTimeSteps) break; // limit document length
        // TODO: do we need to respect masks here? probably it's fine like that
        encoding.get(point(batchIndex), all(), point(t++)).assign(vec.dup());
      }
    }
    return encoding;
  }
  
  public INDArray encodeTarget(List<Sentence> input, List<String> labels) {
    
    INDArray encoding = Nd4j.zeros(DataType.FLOAT, labels.size(), targetEncoder.getEmbeddingVectorSize());
    String label;
    INDArray vec;
    for(int batchIndex = 0; batchIndex < labels.size(); batchIndex++) {
      label = labels.get(batchIndex);
      vec = targetEncoder.encode(label);
      encoding.slice(batchIndex).assign(vec.dup());
    }
    return encoding;
  }

}
