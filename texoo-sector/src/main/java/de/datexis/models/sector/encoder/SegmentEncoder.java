package de.datexis.models.sector.encoder;

import de.datexis.common.Resource;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Document;
import de.datexis.model.Span;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@Deprecated
public class SegmentEncoder extends LookupCacheEncoder {

  public SegmentEncoder() {
    this("SEG");
  }
  
  public SegmentEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(SegmentEncoder.class);
    this.modelAvailable = true;
    totalWords = SegmentTag.Label.values().length;
  }
  
  @Override
  public String getName() {
    return "Segmentation Encoder";
  }

  @Override
  public int getVectorSize() {
    return SegmentTag.Label.values().length;
  }
  
  @Override
  public int getIndex(String word) {
    for(int i=0; i<getVectorSize(); i++) {
      if(SegmentTag.Label.values()[i].toString().equals(word)) return i;
    }
    return -1;
  }

  @Override
  public String getWord(int index) {
    return SegmentTag.Label.values()[index].toString();
  }

  @Override
  public List<String> getWords() {
    return Arrays.asList(SegmentTag.Label.values()).stream().map(l -> l.toString()).collect(Collectors.toList());
  }
  
  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }

  @Override
  public INDArray encode(String word) {
    try { 
      return encode(SegmentTag.Label.valueOf(word));
    } catch(IllegalArgumentException e) {
      return encode(SegmentTag.Label.I);
    }
  }

  public INDArray encode(SegmentTag.Label label) {
    return SegmentTag.getVector(label).transposei();
  }

  @Override
  public void loadModel(Resource modelFile) {
    throw new UnsupportedOperationException("Not applicable to StaticEncoder.");
  }

  @Override
  public void saveModel(Resource modelPath, String name) {
    throw new UnsupportedOperationException("Not applicable to StaticEncoder.");
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    log.debug("Static Encoder does not need to be trained.");
  }
  
}
