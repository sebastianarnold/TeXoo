package de.datexis.cdv.encoder;

import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.impl.BloomEncoder;
import de.datexis.preprocess.IdentityPreprocessor;
import de.datexis.retrieval.tagger.LSTMSentenceTaggerIterator;
import de.datexis.tagger.AbstractMultiDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EntityEncoder extends BloomEncoder {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  /** Used by JSON deserialization */
  public EntityEncoder() {
    super();
    this.preprocessor = new IdentityPreprocessor();
  }
  
  public EntityEncoder(int bitSize, WordHelpers.Language lang) {
    super(bitSize, 5);
    this.preprocessor = new IdentityPreprocessor();
    this.language = lang;
  }
  
  public void trainModel(Resource sentencesTSV) {
    LSTMSentenceTaggerIterator it = new LSTMSentenceTaggerIterator(AbstractMultiDataSetIterator.Stage.ENCODE, null, null, sentencesTSV, "utf-8", WordHelpers.Language.EN, true, 1);
    // Train Bag-of-words and Bloom model
    super.trainModel(it.getLabels(), 1, 1, language);
  }
  
  @Override
  public INDArray encode(String entityId) {
    return super.encode(entityId);
  }
  
  /**
   * Encode a entity into a sparse vector.
   */
  public INDArray decode(String entityId) {
    return super.encode(entityId);
  }
  
  public static Set<String> getDiseaseStopWords() {
    return new TreeSet<>(Arrays.asList(new String[]{",", ".", "(", ")", "[", "]", ";", "'s", "-", "and", "or", "of", "in", "the", "with", "type", "(disorder)", "unspecified",
      "disorder", "disorders", "disease", "diseases", "syndrome", "condition", "conditions", "problem", "problems", "infection", "infections", "illness", "illnesses"}));
  }
  
}
