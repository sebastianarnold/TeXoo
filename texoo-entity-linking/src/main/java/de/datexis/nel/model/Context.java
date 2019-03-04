package de.datexis.nel.model;

import de.datexis.model.Span;
import de.datexis.preprocess.DocumentFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Context is used to encode user's current state in an editor or reading scenario.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class Context extends Span {

  protected final static Logger log = LoggerFactory.getLogger(Context.class);

  protected String text;
  protected int cursor;
  protected String timestamp;
  protected String language;
  
  @Override
  public String getText() {
    return text;
  }
  
  public int getCursor() {
    return cursor;
  }

  public String getTimestamp() {
    return timestamp;
  }
  
  public String getLanguage() {
    if(language != null) return language;
    else return DocumentFactory.getLanguage(text);
  }
  
}
