package de.datexis.retrieval.encoder;

import com.google.common.collect.Lists;
import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.retrieval.tagger.LSTMSentenceTagger;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

/**
 * This Encoder capsules a Sentence Embedding
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LSTMSentenceEncoder extends Encoder {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected LSTMSentenceTagger tagger;
  
  /** used by XML deserialization */
  public LSTMSentenceEncoder() {
    super();
  }
  
  public LSTMSentenceEncoder(LSTMSentenceTagger tagger) {
    this.tagger = tagger;
    this.setId(tagger.getId());
    this.setName(tagger.getName());
  }
  
  public LSTMSentenceTagger getTagger() {
    return tagger;
  }
  
  public void setTagger(LSTMSentenceTagger tagger) {
    this.tagger = tagger;
  }
  
  @Override
  public List<Encoder> getEncoders() {
    return Lists.newArrayList(tagger.getEncoders());
  }
  
  @Override
  public void setEncoders(List<Encoder> encoders) {
    tagger.setEncoders(encoders);
  }
  
  public ComputationGraphConfiguration getGraphConfiguration() {
    return tagger.getGraphConfiguration();
  }
  
  public void setGraphConfiguration(JsonNode conf) {
    tagger.setGraphConfiguration(conf);
  }
  
  @Override
  public void setModelAvailable(boolean modelAvailable) {
    tagger.setModelAvailable(modelAvailable);
  }
  
  @Override
  public boolean isModelAvailable() {
    return tagger.isModelAvailable();
  }
  
  @Override
  public boolean isModelAvailableInChildren() {
    return tagger.isModelAvailableInChildren();
  }
  
  @Override
  public String getModel() {
    return tagger.getModel();
  }
  
  @Override
  protected void setModelFilename(String model) {
    super.setModelFilename(model);
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("Please use LSTMSentenceAnnotator for training.");
  }
  
  @Override
  public void saveModel(Resource dir, String name) throws IOException {
    tagger.saveModel(dir, name);
  }
  
  @Override
  public void loadModel(Resource file) throws IOException {
    tagger.loadModel(file);
  }
  
  @Override
  public long getEmbeddingVectorSize() {
    return getTagger().getEmbeddingLayerSize();
  }
  
  /**
   * Encodes a Sentence into a vector using a forward pass.
   * CAUTION: the sentence will be parsed with an English parser. If you have a tokenized Sentence already, use encode(Sentence)
   * @param sentence
   * @return
   */
  @Override
  public INDArray encode(String sentence) {
    return getTagger().encodeSentence(DocumentFactory.createSentenceFromString(sentence, "EN"));
  }
  
  /**
   * Encodes a Sentence into a vector using a forward pass.
   * @param span the Sentence to encode
   * @return vector for the Sentence
   */
  @Override
  public INDArray encode(Span span) {
    if(span instanceof Sentence) return getTagger().encodeSentence((Sentence) span);
    else throw new UnsupportedOperationException("Not implemented for span type " + span.getClass());
  }
  
}
