package de.datexis.encoder.impl;

import de.datexis.common.WordHelpers;
import de.datexis.encoder.StaticEncoder;
import de.datexis.model.Document;
import de.datexis.model.Span;
import de.datexis.preprocess.DocumentFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Create semantic signatures from context by using a Word2Vec mixture model per Token
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@Deprecated
public class MixtureContextEncoder extends StaticEncoder {

  protected final static Logger log = LoggerFactory.getLogger(MixtureContextEncoder.class);
  
  protected Word2VecEncoder word2vec;
  protected WordHelpers wordHelpers;
  
  public MixtureContextEncoder(String lang) {
    super("CTX");
    word2vec = new Word2VecEncoder();
    wordHelpers = new WordHelpers(WordHelpers.Language.valueOf(lang.toUpperCase()));
  }
  
  public MixtureContextEncoder(String lang, Word2VecEncoder word2vec) {
    super("CTX");
    if (word2vec != null) {
      this.word2vec = word2vec;
      wordHelpers = new WordHelpers(WordHelpers.Language.valueOf(lang.toUpperCase()));
    } else log.error("Could not load word2vec / word2vec == null");
  }
  
  @Override
  public int getVectorSize() {
    return word2vec.getVectorSize();
  }
  
  @Override
  public INDArray encode(String text) {
    Document doc = DocumentFactory.fromText(text);
    return encode(doc);
  }
  
  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }
  
  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray avg = Nd4j.create(getVectorSize(), 1);
    INDArray vec;
    int i = 0;
    for(Span s : spans) {
      if(wordHelpers.isStopWord(s.getText())) continue;
      vec = word2vec.encode(s.getText());
      if(vec != null) {
        avg.addi(vec);
        i++;
      }
    }
    return avg.divi(i);//avg.mean(0);
  }

  @Override
  public String getName() {
    return "Mixture Context Encoder";
  }
  
}
