package de.datexis.model.annotation;

import de.datexis.model.Annotation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple Annotation used to attach Terms to documents, e.g. Titles, Labels, IndexTerms, etc.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class TermAnnotation extends Annotation {

  protected final static Logger log = LoggerFactory.getLogger(TermAnnotation.class);
  
  public TermAnnotation(Source source, String term) {
    super(source, term);
  }

}
