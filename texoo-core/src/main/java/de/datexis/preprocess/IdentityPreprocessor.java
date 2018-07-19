package de.datexis.preprocess;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class IdentityPreprocessor implements TokenPreProcess {

  protected final static Logger log = LoggerFactory.getLogger(IdentityPreprocessor.class);

  public IdentityPreprocessor() {
  }

  @Override
  public String preProcess(String token) {
    return token;
  }

}
