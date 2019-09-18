package de.datexis.encoder;

import com.google.common.collect.Lists;
import de.datexis.annotator.Annotator;
import de.datexis.tagger.EmptyTagger;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

/**
 * A wrapper that holds a single encoder so it can be saved as an XML configuration.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EncoderAnnotator extends Annotator {
  
  public EncoderAnnotator() {}
  
  public EncoderAnnotator(Encoder encoder) {
    this.tagger = new EmptyTagger();
    this.tagger.setId("ENC");
    this.tagger.setEncoders(Lists.newArrayList(encoder));
    this.addComponent(encoder);
  }
  
  @JsonIgnore
  public Encoder getEncoder() {
    return tagger.getEncoders().get(0);
  }
  
}
