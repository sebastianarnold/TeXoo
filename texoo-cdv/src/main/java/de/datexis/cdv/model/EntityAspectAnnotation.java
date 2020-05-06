package de.datexis.cdv.model;

import de.datexis.model.impl.PassageAnnotation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EntityAspectAnnotation extends PassageAnnotation {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected String entity;
  protected String entityId;
  protected String aspect;
  protected String heading;
  
  /**  Used for JSON Deserialization */
  protected EntityAspectAnnotation() {}
  
  public EntityAspectAnnotation(Source source) {
    super(source);
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
  
  public String getHeading() {
    return heading;
  }
  
  public void setHeading(String heading) {
    this.heading = heading;
  }
  
  public String toString() {
    return String.format("Passage<'%s' (%s), '%s'>", getEntity(), getEntityId(), getAspect());
  }
  
}
