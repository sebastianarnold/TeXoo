package de.datexis.models.sector.encoder;

import de.datexis.encoder.impl.BagOfWordsEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Wrapper for Bag-Of-Words Headings
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class HeadingEncoder extends BagOfWordsEncoder {

  protected final static Logger log = LoggerFactory.getLogger(HeadingEncoder.class);
public static final String ID = "HL";
  
  public HeadingEncoder() {
    super(ID);
  }
  
}
