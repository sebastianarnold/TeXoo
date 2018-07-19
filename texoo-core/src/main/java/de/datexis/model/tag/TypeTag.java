package de.datexis.model.tag;

import java.util.Objects;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The Type of a Token as String.
 * @author sarnold
 */
public class TypeTag implements Tag {

  private static final Logger log = LoggerFactory.getLogger(TypeTag.class);

  public static class Type {
		public static final String GENERIC = "GENERIC";
    public static final String NOUNPHRASE = "NP";
    public static final String NOUN = "NOUN";
    public static final String PER = "PER";
    public static final String ORG = "ORG";
    public static final String LOC = "LOC";
    public static final String MISC = "MISC";
	}
  
  protected String type;
  protected INDArray vector;
  protected double confidence = 0.;
 
  @Override
  public int getVectorSize() {
    return 0;
  }
  
  @Override
  public INDArray getVector() {
    return vector;
  }
  
  @Override
  public double getConfidence() {
    return confidence;
  }
  
  public TypeTag setConfidence(double confidence) {
    this.confidence = confidence;
    return this;
  }
  
  public String getType() {
    return type;
  }
  
  @Override
  public String toString() {
    return type;
  }

  @Override
  public String getTag() {
    return type.toString();
  }
  
  @Override
  public String getTag(int index) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int hashCode() {
    int hash = 3;
    hash = 79 * hash + Objects.hashCode(this.type);
    return hash;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    final TypeTag other = (TypeTag) obj;
    if (!Objects.equals(this.type, other.type)) {
      return false;
    }
    return true;
  }
  
}
