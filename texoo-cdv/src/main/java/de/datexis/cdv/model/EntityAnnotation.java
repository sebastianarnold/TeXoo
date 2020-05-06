package de.datexis.cdv.model;

import de.datexis.model.Annotation;
import de.datexis.model.impl.PassageAnnotation;

/**
 * An Annotation that describes a latent topic over the course of a document.
 * E.g. "this passage is about entity X".
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EntityAnnotation extends PassageAnnotation {
  // this class is only used to distinguish dimensions
  public EntityAnnotation(Annotation.Source source) {
    super(source);
  }
  protected EntityAnnotation() {}
}
