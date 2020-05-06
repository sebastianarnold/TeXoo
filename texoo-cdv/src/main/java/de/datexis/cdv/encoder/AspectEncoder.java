package de.datexis.cdv.encoder;

import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.preprocess.AspectPreprocessor;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.impl.BloomEncoder;
import de.datexis.retrieval.tagger.LSTMSentenceTaggerIterator;
import de.datexis.tagger.AbstractMultiDataSetIterator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AspectEncoder extends BloomEncoder {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected int minWordFreq;
  protected int minWordLength = 3;
  
  /** Used by JSON deserialization */
  public AspectEncoder() {
    super();
    this.preprocessor = new AspectPreprocessor();
  }
  
  public AspectEncoder(int bitSize, WordHelpers.Language lang, int minWordFreq) {
    super(bitSize, 5);
    this.preprocessor = new AspectPreprocessor();
    this.language = lang;
    this.minWordFreq = minWordFreq;
  }
  
  public int getMinWordFreq() {
    return minWordFreq;
  }
  
  public void setMinWordFreq(int minWordFreq) {
    this.minWordFreq = minWordFreq;
  }
  
  public int getMinWordLength() {
    return minWordLength;
  }
  
  public void setMinWordLength(int minWordLength) {
    this.minWordLength = minWordLength;
  }
  
  public void trainModel(Resource sentencesTSV) {
    LSTMSentenceTaggerIterator it = new LSTMSentenceTaggerIterator(AbstractMultiDataSetIterator.Stage.ENCODE, null, null, sentencesTSV, "utf-8", WordHelpers.Language.EN, true, 1);
    List<String> labels = it.getLabels();
    labels = labels.stream()
      .map(l -> l.replaceAll("/", " ")) // replace slashes to trim words
      .collect(Collectors.toList());
    // Train Bag-of-words and Bloom model
    super.trainModel(labels, minWordFreq, minWordLength, language);
  }
  
  @Override
  public INDArray encode(String heading) {
    String[] headings = heading.split(AspectIndex.HEADING_SEPARATOR_REGEX);
    INDArray sum = Nd4j.zeros(DataType.FLOAT, getEmbeddingVectorSize(), 1);
    INDArray vec;
    for(String h : headings) {
      vec = super.encode(h);
      if(vec != null) sum.addi(vec);
    }
    return sum;
  }
  
  /**
   * Encode a entity into a sparse vector.
   */
  public INDArray decode(String heading) {
    return encode(heading);
  }
  
}
