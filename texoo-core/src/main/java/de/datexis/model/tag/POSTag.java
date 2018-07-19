package de.datexis.model.tag;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Objects;

/**
 * A part-of-speech Tag for a single Token.
 * @author sarnold
 */
public class POSTag implements Tag {
  
  protected String value;
  
  public POSTag() {
    this("");
  }
  
  public POSTag(String value) {
    this.value = value;
  }
  
  public String getValue() {
    return value;
  }

  @Override
  public INDArray getVector() {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public int getVectorSize() {
    throw new UnsupportedOperationException("Not implemented yet.");
  }
  
  @Override
  public String getTag() {
    return value;
  }
  
  @Override
  public String getTag(int index) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public double getConfidence() {
    return 0.;
  }

  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof POSTag)) {
      return false;
    }
    POSTag posTag = (POSTag) o;
    return Objects.equals(getValue(), posTag.getValue());
  }

  @Override
  public int hashCode() {
    return Objects.hash(getValue());
  }
}
