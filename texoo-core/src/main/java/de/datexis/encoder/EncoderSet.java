package de.datexis.encoder;

import de.datexis.model.Span;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.LoggerFactory;

/**
 * A set of Encoders for vectors that will be concatenated as input
 * @author sarnold
 */
// TODO: Extends AbstractEncoder would be handy
public class EncoderSet implements Iterable<AbstractEncoder> {
  
  protected static final org.slf4j.Logger log = LoggerFactory.getLogger(EncoderSet.class);
  
  protected List<AbstractEncoder> encoders;
  protected int size;
  
  public EncoderSet(AbstractEncoder... encoders) {
    this.encoders = new ArrayList<>(encoders.length);
    this.size = 0;
    for(AbstractEncoder enc : encoders) {
      addEncoder(enc);
    }
  }
  
  public final void addEncoder(AbstractEncoder e) {
    encoders.add(e);
    if(e.getVectorSize() == 0) log.warn("Adding uninitialized AbstractEncoder " + e.getName());
    this.size += e.getVectorSize();
  }
  
  /**
   * Recalculates vector size in case one AbstractEncoder has changed
   */
  public void updateVectorSize() {
    this.size = 0;
    for(AbstractEncoder enc : this.encoders) {
      this.size += enc.getVectorSize();
    }
  }
  
  public int getVectorSize() {
    return size;
  }
  
  public Iterable<AbstractEncoder> iterable() {
    return encoders;
  }

  @Override
  public Iterator<AbstractEncoder> iterator() {
    return encoders.iterator();
  }
  
  /**
   * Encodes a given String using all Encoders. Does not save the intermediate results to the Tokens.
   */
  public INDArray encode(String word) {
    INDArray result = Nd4j.create(getVectorSize());
    int i = 0;
    for(AbstractEncoder enc : encoders) {
      final INDArray vec = enc.encode(word);
      result.get(NDArrayIndex.interval(i, i + enc.getVectorSize())).assign(vec);
      i += enc.getVectorSize();
    }
    return result;
  }
  
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray result = Nd4j.create(getVectorSize());
    int i = 0;
    for(AbstractEncoder enc : encoders) {
      final INDArray vec = enc.encode(spans);
      result.get(NDArrayIndex.interval(i, i + enc.getVectorSize())).assign(vec);
      i += enc.getVectorSize();
    }
    return result;
  }
  
  @Override
  public String toString() {
    return encoders.stream().map(e -> e.getId()).collect(Collectors.joining("-"));
  }
  
}
