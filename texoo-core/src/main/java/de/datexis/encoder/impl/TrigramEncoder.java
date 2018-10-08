package de.datexis.encoder.impl;

import org.slf4j.LoggerFactory;

/**
 * More readable Version of our heavily-used Trigram-AbstractEncoder
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class TrigramEncoder extends LetterNGramEncoder {
  
  public TrigramEncoder() {
    super("TRI");
    log = LoggerFactory.getLogger(TrigramEncoder.class);
    this.setN(3);
  }
  
  public TrigramEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(TrigramEncoder.class);
    this.setN(3);
  }
  
}
