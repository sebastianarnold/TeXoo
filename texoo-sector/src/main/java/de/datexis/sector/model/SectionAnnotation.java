package de.datexis.sector.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.model.tag.TFTag;
import de.datexis.sector.encoder.ClassTag;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.Collectors;

/**
 * An Annotation used to attach a phrase (e.g. sentences) to documents, e.g. questions, summarizations, etc.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonPropertyOrder({"class", "source", "begin", "length", "sectionHeading", "sectionLabel" })
@JsonIgnoreProperties({"confidence", "text"})
public class SectionAnnotation extends PassageAnnotation {

  protected final static Logger log = LoggerFactory.getLogger(SectionAnnotation.class);

  protected String type;
  protected String normalizedLabel;
  
  public static enum Field { HEADING, LABEL };
  
  /**  Used for JSON Deserialization */
  protected SectionAnnotation() {}
  
  public SectionAnnotation(Annotation.Source source) {
    super(source);
  }
  
  public SectionAnnotation(Annotation.Source source, String type, String sectionHeading) {
    super(source);
    this.type = type;
    this.label = sectionHeading;
  }
  
  /**
   * @return the normalized section label
   */
  public String getSectionLabel() {
    return normalizedLabel;
  }
  
  public void setSectionLabel(String normalizedLabel) {
    this.normalizedLabel = normalizedLabel;
  }
  
  /**
   * @return the section heading
   */
  public String getSectionHeading() {
    return label;
  }

  public void setSectionHeading(String sectionHeading) {
    this.label = sectionHeading;
  }
  
  @JsonIgnore
  public String getSectionLabelOrHeading() {
    if(normalizedLabel != null && !normalizedLabel.isEmpty()) return normalizedLabel;
    if(label != null && !label.isEmpty()) return label;
    else return "";
  }
  
  public String getAnnotation(Field field) {
    if(field.equals(Field.LABEL))  return normalizedLabel;
    else return label;
  }
  
  @JsonIgnore
  public String getType() {
    return type;
  }
 
  /**
   * Returns TRUE, if this Annotation matches the boundaries and class of another annotation
   * @param other
   * @return 
   */
  public boolean matches(SectionAnnotation other) {
    if(!this.getSectionLabel().equals(other.getSectionLabel())) return false;
    int p1 = this.getBegin();
    int p2 = other.getBegin();
    int e1 = p1 + this.getLength() - 1;
    int e2 = p2 + other.getLength() - 1;
    return (p1<=p2 && p2<=e1) || (p1<=e2 && e2<=e1) ||
           (p2<=p1 && p1<=e2) || (p2<=e1 && e1<=e2);
  }
  
  /**
   * Two annotations are equal, iff begin and length are equal
   * @param obj
   * @return 
   */
  @Override
  public boolean equals(Object obj) {
    if(this == obj) return true;
    if(obj == null) return false;
    if(getClass() != obj.getClass()) return false;
    final SectionAnnotation other = (SectionAnnotation) obj;
    if(this.begin != other.getBegin()) return false;
    if(this.end != other.getEnd()) return false;
    // if(this.source != other.source) return false; // DON'T include source to compare GOLD and PRED
    if(getSectionLabel() == null) {
			if(other.getSectionLabel() != null) return false;
		} else if(!getSectionLabel().equals(other.getSectionLabel())) return false;
    return true;
  }
  
  @Deprecated
  public static void createClassTagsFromAnnotations(Document doc, Source source) {
    boolean flip = true;
    for(SectionAnnotation ann : doc.getAnnotations(source, SectionAnnotation.class)) {
      boolean begin = true;
      for(Sentence s : doc.streamSentencesInRange(ann.getBegin(), ann.getEnd(), false).collect(Collectors.toList())) {
        s.putTag(source, new ClassTag(ann.getSectionLabel(), Nd4j.create(0)));
        s.putTag(source, flip ? TFTag.T() : TFTag.F());
        s.putTag(source, begin ? BIO2Tag.B() : BIO2Tag.I());
        begin = false;
      }
      flip = !flip;
    }
  }

}
