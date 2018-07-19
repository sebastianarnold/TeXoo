package de.datexis.models.sector.encoder;

import de.datexis.encoder.Encoder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.tag.Tag;
import de.datexis.models.sector.model.SectionAnnotation;
import java.util.Iterator;
import java.util.Objects;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Tag that is used to label a single Class.
 * @author sarnold
 */
@Deprecated
public class SegmentTag implements Tag {

  private static final Logger log = LoggerFactory.getLogger(SegmentTag.class);

  public SegmentTag(Label label) {
    this.label = label;
    this.vector = null;
  }
  
  public SegmentTag(INDArray predicted, boolean storeVector) {
    this.label = max(predicted);
    this.vector = storeVector ? predicted : null;
    this.confidence = predicted.getDouble(this.label.ordinal());
  }
  
  /**
   * The Label of the token in BIO2 format, used for sequence learning
   * B = Begin Segment
   * I = Inside Segment
   */
  public static enum Label {
    B, I
  }
  
  public static Enum<?>[] getLabels() {
    return Label.values();
  }
  
  public static SegmentTag B() { return new SegmentTag(Label.B); }
  public static SegmentTag I() { return new SegmentTag(Label.I); }
  
  protected final Label label;
  protected final INDArray vector;
  protected double confidence = 0.;
 
  public Label get() {
    return label;
  }
  
  public static INDArray getVector(Label tag) {
		switch(tag) {
		case B:
			return Nd4j.create(new double[] { 1, 0 });
		default:
			return Nd4j.create(new double[] { 0, 1 });
		}
	}
  
  public boolean isB() {
    return label.equals(Label.B);
  }
  
  public boolean isI() {
    return label.equals(Label.I);
  }
  
  /**
   * Returns the highest-scored label
   * @param predicted
   * @return 
   */
  public static Label max(INDArray predicted) {
    double max = predicted.getDouble(0);
    int currMax = 0;
    for(int col = 1; col < predicted.length(); col++) {
      if(predicted.getDouble(col) >= max) {
        max = predicted.getDouble(col);
        currMax = col;
      }
    }
    if(currMax == 0) return Label.B;
    else return Label.I;
  }
  
  @Override
  public int getVectorSize() {
    return 2;
  }
  
  @Override
  public INDArray getVector() {
    if(vector != null) return vector;
    else return getVector(label);
  }
  
  @Override
  public double getConfidence() {
    return confidence;
  }
  
  public SegmentTag setConfidence(double confidence) {
    this.confidence = confidence;
    return this;
  }
  
  @Override
  public String toString() {
    return label.toString();
  }

  @Override
  public String getTag() {
    return label.toString();
  }
  
  @Override
  public String getTag(int index) {
    return Label.values()[index].toString();
  }

  @Override
  public int hashCode() {
    int hash = 3;
    hash = 79 * hash + Objects.hashCode(this.label);
    return hash;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    final SegmentTag other = (SegmentTag) obj;
    if (this.label != other.label) {
      return false;
    }
    return true;
  }
  
  public static class Factory {
  
    protected final SegmentEncoder encoder;
    
    public Factory(SegmentEncoder encoder) {
      this.encoder = encoder;
    }
    
    public SegmentTag create(Label tag) {
      return new SegmentTag(tag);
    }
    
    public SegmentTag create(INDArray prediction) {
      return new SegmentTag(prediction, true);
    }
    
    public void attachFromSectionAnnotations(Document doc, Annotation.Source source) {
      for(SectionAnnotation ann : doc.getAnnotations(source, SectionAnnotation.class)) {
        Iterator<Sentence> it = doc.streamSentencesInRange(ann.getBegin(), ann.getEnd(), false).iterator();
        if(it.hasNext()) {
          it.next().putTag(source, create(encoder.encode(Label.B)));
          while(it.hasNext()) {
            it.next().putTag(source, create(encoder.encode(Label.I)));
          }
        }
      }
      doc.setTagAvailable(source, SegmentTag.class, true);
    }
    
    public void attachFromSentenceVectors(Document doc, Class<? extends Encoder> encoder, Annotation.Source source) {
      for(Sentence s : doc.getSentences()) {
        s.putTag(source, create(s.getVector(encoder)));
      }
      doc.setTagAvailable(source, SegmentTag.class, true);
    }
    
  }
  
}
