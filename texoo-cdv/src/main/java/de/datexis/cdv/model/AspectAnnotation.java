package de.datexis.cdv.model;

import de.datexis.model.Annotation;
import de.datexis.model.impl.PassageAnnotation;

/**
 * An Annotation that describes a latent aspect over the course of a document.
 * E.g. "this passage discusses aspect Y".
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AspectAnnotation extends PassageAnnotation {
  // this class is only used to distinguish dimensions
  public AspectAnnotation(Annotation.Source source) {
    super(source);
  }
  protected AspectAnnotation() {}
}
