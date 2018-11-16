package de.datexis.encoder;

import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Example implementation of an 128-dim Encoder / 8-class Decoder
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ExampleEncoder implements IEncoder, IDecoder {

  @Override
  public long getEmbeddingVectorSize() {
    return 128L;
  }

  @Override
  public INDArray encode(String word) {
    return Nd4j.ones(getEmbeddingVectorSize(), 1);
  }

  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }

  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray avg = Nd4j.create(getEmbeddingVectorSize(), 1);
    INDArray vec;
    int i = 0;
    for(Span s : spans) {
      vec = encode(s.getText());
      if(vec != null) {
        avg.addi(vec);
        i++;
      }
    }
    return avg.divi(i);
  }

  @Override
  public long getOutputVectorSize() {
    return 8L;
  }

  @Override
  public INDArray decode(String word) {
    return Nd4j.ones(getOutputVectorSize(), 1);
  }

  @Override
  public INDArray decode(Span span) {
    return decode(span.getText());
  }

  @Override
  public INDArray decode(Iterable<? extends Span> spans) {
    INDArray avg = Nd4j.create(getOutputVectorSize(), 1);
    INDArray vec;
    int i = 0;
    for(Span s : spans) {
      vec = decode(s.getText());
      if(vec != null) {
        avg.addi(vec);
        i++;
      }
    }
    return avg.divi(i);
  }
  
}
