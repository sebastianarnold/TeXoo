package de.datexis.model.annotation;

import de.datexis.model.Annotation;
import de.datexis.model.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Annotation used to attach a phrase (e.g. sentences) to documents, e.g. questions, summarizations, etc.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class PhraseAnnotation extends Annotation {

  protected final static Logger log = LoggerFactory.getLogger(PhraseAnnotation.class);
  
  protected Document phrase;
  
  public PhraseAnnotation(Source source, Document phrase) {
    super(source, "");
    this.phrase = phrase;
  }

  @Override
  public String getText() {
    return phrase.getText();
  }
  
  public Document getPhrase() {
    return phrase;
  }
  
}
