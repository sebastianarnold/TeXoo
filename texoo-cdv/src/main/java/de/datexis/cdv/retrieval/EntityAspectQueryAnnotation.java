package de.datexis.cdv.retrieval;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import de.datexis.model.Annotation;
import de.datexis.model.Query;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties({"begin", "length", "confidence"})
public class EntityAspectQueryAnnotation extends Annotation {
  
  protected final static Logger log = LoggerFactory.getLogger(EntityAspectQueryAnnotation.class);
  
  protected String entity;
  protected String entityId;
  protected String aspect;
  protected String aspectHeading;
  
//  protected String focusType;
//  protected Set<String> focusSynonyms;
//
//  protected String aspectDescription;
//  protected Set<String> aspectSynonyms;
  
  /** default constructor for JSON deserialization */
  protected EntityAspectQueryAnnotation() {};
  
  /**
   * Create a new Document from plain text
   */
  public static Query createQuery(String entity, String aspect) {
    Query result = new Query();
    EntityAspectQueryAnnotation ann = new EntityAspectQueryAnnotation(entity,aspect);
    result.addAnnotation(ann);
    result.setText(ann.getText());
    return result;
  }
  
  public EntityAspectQueryAnnotation(String entity, String aspect) {
    super();
    this.entity = entity;
    this.aspect = aspect;
  }
  
  @Override
  @JsonIgnore
  public Source getSource() {
    return source;
  }
  
  public String getEntity() {
    return entity;
  }
  
  public void setEntity(String entity) {
    this.entity = entity;
  }
  
  public String getEntityId() {
    return entityId;
  }
  
  public void setEntityId(String entityId) {
    this.entityId = entityId;
  }
  
  public String getAspect() {
    return aspect;
  }
  
  public void setAspect(String aspect) {
    this.aspect = aspect;
  }
  
  public String getAspectHeading() {
    return aspectHeading;
  }
  
  public void setAspectHeading(String aspectHeading) {
    this.aspectHeading = aspectHeading;
  }
  
  @JsonIgnore
  public boolean hasEntity() {
    return entity != null && !entity.trim().isEmpty();
  }
  
  @JsonIgnore
  public boolean hasAspect() {
    return aspect != null && !aspect.trim().isEmpty();
  }
  
  @JsonIgnore
  public boolean hasEntityAndAspect() {
    return hasEntity() && hasAspect();
  }
  
  @JsonIgnore
  public boolean isEmpty() {
    return !(hasEntity() || hasAspect());
  }
  
  @Override
  @JsonIgnore
  public String getText() {
    if(hasEntityAndAspect()) {
      return "[" + aspect + "] for [" + entity + "]";
    } else if(hasEntity()) {
      return "entity: [" + entity + "]";
    } else if(hasAspect()) {
      return "aspect: [" + aspect + "]";
    } else {
      return "";
    }
  }
  
  /**
   * Returns TRUE, if this Annotation matches aspect and entity
   * @param other
   * @return
   */
  public boolean matches(Annotation other) {
    if(!(other instanceof EntityAspectQueryAnnotation)) return false;
    return this.getAspect().equals(((EntityAspectQueryAnnotation) other).getAspect()) &&
      (
        (this.getEntityId() != null && ((EntityAspectQueryAnnotation) other).getEntityId() != null  && this.getEntityId().equals(((EntityAspectQueryAnnotation) other).getEntityId())) ||
        this.getEntity().equals(((EntityAspectQueryAnnotation) other).getEntity())
      );
      
  }
  
}
