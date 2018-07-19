package de.datexis.model.tag;

import de.datexis.model.Annotation.Source;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * A Tag that is used to label a single Token in BIOES format, used for sequence learning.
 * @author sarnold
 */
public class TFTag implements Tag {

  private static final Logger log = LoggerFactory.getLogger(TFTag.class);

  /**
   * The Label of the token in BIOES format, used for sequence learning
   * T = True
   * F = False
   */
  public static enum Label {
    T, F
  }

  public static Enum<?>[] getLabels() {
    return Label.values();
  }

  public static TFTag T() { return new TFTag(Label.T); }
  public static TFTag F() { return new TFTag(Label.F); }

  protected final Label label;
  protected final String type;
  protected final INDArray vector;
  protected double confidence = 0.;

  public TFTag() {
    this(Label.F, null);
  }

  public TFTag(Label tag, String type) {
    this.label = tag;
    this.type = type;
    this.vector = null;
  }

  public TFTag(INDArray predicted, String type, boolean storeVector) {
    this.label = max(predicted);
    this.type = type;
    this.vector = storeVector ? predicted.detach() : null;
    this.confidence = predicted.getDouble(this.label.ordinal());
  }

  public TFTag(Label tag) {
    this(tag, tag.equals(Label.F) ? null : GENERIC);
  }

  public TFTag(Label tag, INDArray predicted, boolean storeVector) {
    this.label = tag;
    this.type = tag.equals(Label.F) ? null : GENERIC;
    this.vector = storeVector ? predicted.detach() : null;
    this.confidence = predicted.getDouble(this.label.ordinal());
  }

  public TFTag(INDArray predicted, boolean StoreVector) {
    this(predicted, GENERIC, StoreVector);
  }

  public Label get() {
    return label;
  }

  public static INDArray getVector(Label tag) {
		switch(tag) {
		case T:
			return Nd4j.create(new double[] { 1, 0 });
    default:
    case F:
			return Nd4j.create(new double[] { 0, 1 });
		}
	}

  public boolean isT() {
    return label.equals(Label.T);
  }

  public boolean isF() {
    return label.equals(Label.F);
  }

  @Override
  public double getConfidence() {
    return this.confidence;
  }

  public TFTag setConfidence(double confidence) {
    this.confidence = confidence;
    return this;
  }

  /**
   * Returns the highest-scored label
   * @param predicted
   * @return
   */
  public static Label max(INDArray predicted) {
    double max = predicted.getDouble(0);
    int currMax = 0;
    for(int col = 1; col < predicted.length(); col++) {
      if(predicted.getDouble(col) >= max) {
        max = predicted.getDouble(col);
        currMax = col;
      }
    }
    return index(currMax);
  }

  /**
   * Return the n-th Label
   * @param x
   * @return
   */
  public static Label index(int x) {
    if(x == 0) return Label.T;
    else return Label.F;
  }

  @Override
  public int getVectorSize() {
    return 2;
  }

  /**
   * @return The predicted Vector for this Ta
   */
  @Override
  public INDArray getVector() {
    if(vector != null) return vector;
    else return getVector(label);
  }

  public String getType() {
    return type;
  }

  @Override
  public String toString() {
    if(type == null) return label.toString();
    else return label.toString() + "-" + type;
  }

  @Override
  public String getTag() {
    return label.toString();
  }

  @Override
  public String getTag(int index) {
    return Label.values()[index].toString();
  }

  @Override
  public int hashCode() {
    int hash = 7;
    hash = 47 * hash + Objects.hashCode(this.label);
    hash = 47 * hash + Objects.hashCode(this.type);
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
    final TFTag other = (TFTag) obj;
    if (!Objects.equals(this.type, other.type)) {
      return false;
    }
    if (this.label != other.label) {
      return false;
    }
    return true;
  }

  private static INDArray getLabelVector(ArrayList<Token> tokens, int pos, Source source) {
    if(pos < 0 || pos >= tokens.size()) return TFTag.getVector(TFTag.Label.F);
    else return tokens.get(pos).getTag(source, TFTag.class).getVector();
  }

  private static Label getLabel(ArrayList<Token> tokens, int pos, Source source) {
    if(pos < 0 || pos >= tokens.size()) return TFTag.Label.F;
    else return tokens.get(pos).getTag(source, TFTag.class).get();
  }
  
}
