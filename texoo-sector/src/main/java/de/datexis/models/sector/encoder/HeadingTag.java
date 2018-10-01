package de.datexis.models.sector.encoder;

import de.datexis.encoder.Encoder;
import de.datexis.encoder.impl.BagOfWordsEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.tag.Tag;
import de.datexis.models.sector.model.SectionAnnotation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class HeadingTag implements Tag {

  protected final static Logger log = LoggerFactory.getLogger(HeadingTag.class);

  protected final String label;
  protected final INDArray vector;
  protected double confidence = 0.;
  
  protected HeadingTag(String label, INDArray vector) {
    this.label = label;
    this.vector = vector.detach();
  }
  
  @Override
  public String getTag() {
    return label;
  }

  @Override
  public String getTag(int index) {
    throw new UnsupportedOperationException("Not possible without encoder.");
  }
  
  @Override
  public int getVectorSize() {
    return (int) vector.length();
  }

  @Override
  public INDArray getVector() {
    return vector.detach();
  }

  @Override
  public double getConfidence() {
    return this.confidence;
  }
  
  public void setConfidence(double confidence) {
    this.confidence = confidence;
  }
  
  public static class Factory {
  
    protected final BagOfWordsEncoder encoder;
    
    public Factory(BagOfWordsEncoder encoder) {
      this.encoder = encoder;
    }
    
    public HeadingTag create(String heading) {
      return new HeadingTag(heading, encoder.encode(heading));
    }
    
    public HeadingTag create(INDArray prediction) {
      return new HeadingTag(encoder.getNearestNeighbour(prediction), prediction);
    }
    
    public void attachFromSectionAnnotations(Document doc, Annotation.Source source) {
      for(SectionAnnotation ann : doc.getAnnotations(source, SectionAnnotation.class)) {
        String title = ann.getSectionHeading();
        INDArray vec = encoder.encode(title);
        doc.streamSentencesInRange(ann.getBegin(), ann.getEnd(), false).forEach(
          s -> s.putTag(source, new HeadingTag(title, vec))
        );
      }
      doc.setTagAvailable(source, HeadingTag.class, true);
    }
    
    public void attachFromSentenceVectors(Document doc, Class<? extends Encoder> encoder, Annotation.Source source) {
      for(Sentence s : doc.getSentences()) {
        s.putTag(source, create(s.getVector(encoder)));
      }
      doc.setTagAvailable(source, HeadingTag.class, true);
    }
    
  }

}
