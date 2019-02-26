package de.datexis.model.impl;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import de.datexis.model.Annotation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Annotation used to point at a passage in a document
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonPropertyOrder({"class", "source", "begin", "length", "id" })
@JsonIgnoreProperties({"confidence", "text"})
public class PassageAnnotation extends Annotation {

  protected final static Logger log = LoggerFactory.getLogger(PassageAnnotation.class);
  
  /**
   * The ID of this Annotation
   */
  private String id = null;
  
  /**  Used for JSON Deserialization */
  protected PassageAnnotation() {}
  
  public PassageAnnotation(Source source) {
    super(source, "");
  }
  
  public PassageAnnotation(Source source, String type, String sectionHeading) {
    super(source, "");
  }
  
  public void setId(String id) {
    this.id = id;
  }
  
  @JsonInclude(JsonInclude.Include.NON_NULL)
  public String getId() {
    return this.id;
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
    final PassageAnnotation other = (PassageAnnotation) obj;
    if(this.begin != other.getBegin()) return false;
    if(this.end != other.getEnd()) return false;
    // if(this.source != other.source) return false; // DON'T include source to compare GOLD and PRED
    if(id == null) {
			if(other.getId() != null) return false;
		} else if(!id.equals(other.getId())) return false;
    return true;
  }

}
