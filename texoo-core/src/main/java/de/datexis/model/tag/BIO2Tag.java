package de.datexis.model.tag;

import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import java.util.Iterator;
import java.util.Objects;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Tag that is used to label a single Token in BIO2 format, used for sequence learning.
 * @author sarnold
 */
public class BIO2Tag implements Tag {

  private static final Logger log = LoggerFactory.getLogger(BIO2Tag.class);
  
  /**
   * The Label of the token in BIO2 format, used for sequence learning
   * B = Begin Annotation
   * I = Inside Annotation
   * O = Outside Annotation
   */
  public static enum Label {
    B, I, O
    /* is this safer than ordinal() ?
    B(0), I(1), O(2);
    private final int index;
    Label(int index) { this.index = index; }
    public int getIndex() { return index; }*/
  }
  
  public static Enum<?>[] getLabels() {
    return Label.values();
  }
  
  public static BIO2Tag B() { return new BIO2Tag(Label.B); }
  public static BIO2Tag I() { return new BIO2Tag(Label.I); }
  public static BIO2Tag O() { return new BIO2Tag(Label.O); }
  
  protected final Label label;
  protected String type;
  protected final INDArray vector;
  protected double confidence = 0.;
  
  public BIO2Tag() {
    this(Label.O, null);
  }
  
  public BIO2Tag(Label label, String type) {
    this.label = label;
    this.type = type;
    this.vector = null;
  }
  
  public BIO2Tag(INDArray predicted, String type, boolean storeVector) {
    this.label = max(predicted);
    this.type = type;
    this.vector = storeVector ? predicted.detach() : null;
    this.confidence = predicted.getDouble(this.label.ordinal());
  }
  
  public BIO2Tag(Label tag) {
    this(tag, tag.equals(Label.O) ? null : GENERIC);
  }
  
  public BIO2Tag(INDArray predicted, boolean storeVector) {
    this(predicted, GENERIC, storeVector);
  }
  
  public Label get() {
    return label;
  }
  
  public static INDArray getVector(Label tag) {
		switch(tag) {
		case B:
			return Nd4j.create(new double[] { 1, 0, 0 });
		case I:
			return Nd4j.create(new double[] { 0, 1, 0 });
		default:
			return Nd4j.create(new double[] { 0, 0, 1 });
		}
	}
  
  public boolean isB() {
    return label.equals(Label.B);
  }
  
  public boolean isI() {
    return label.equals(Label.I);
  }
  
  public boolean isO() {
    return label.equals(Label.O);
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
    if(currMax == 0) return Label.B;
    else if(currMax == 1) return Label.I;
    else return Label.O;
  }
  
  @Override
  public int getVectorSize() {
    return 3;
  }
  
  @Override
  public INDArray getVector() {
    if(vector != null) return vector;
    else return getVector(label);
  }
  
  @Override
  public double getConfidence() {
    return confidence;
  }
  
  public BIO2Tag setConfidence(double confidence) {
    this.confidence = confidence;
    return this;
  }
  
  public BIO2Tag setType(String type) {
    this.type = type;
    return this;
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
    int hash = 3;
    hash = 79 * hash + Objects.hashCode(this.label);
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
    final BIO2Tag other = (BIO2Tag) obj;
    if (!Objects.equals(this.type, other.type)) {
      return false;
    }
    if (this.label != other.label) {
      return false;
    }
    return true;
  }
  
  /**
   * Converts BIO2 Tags to BIOES Tags
   * requires: BIO2Tag.class on Token.class
   * attaches: BIOESTag.class to Token.class
   * @param data
   * @param source 
   */
  public static void convertToBIOES(Dataset data, Annotation.Source source) {
    for(Document doc : data.getDocuments()) {
      for(Sentence sent : doc.getSentences()) {
        convertToBIOES(sent, source);
      }
    }
  }
  
  /**
   * Converts BIO2 Tags to BIOES Tags
   * requires: BIO2Tag.class on Token.class
   * attaches: BIOESTag.class to Token.class
   * @param data
   * @param source 
   */
  public static void convertToBIOES(Document doc, Annotation.Source source) {
    for(Sentence sent : doc.getSentences()) {
      convertToBIOES(sent, source);
    }
  }
  
  /**
   * Converts BIO2 Tags to BIOES Tags
   * requires: BIO2Tag.class on Token.class
   * attaches: BIOESTag.class to Token.class
   * @param sent
   * @param source 
   */
  public static synchronized void convertToBIOES(Sentence sent, Annotation.Source source) {
    Token l = new Token("");
    Token t = new Token("");
    Iterator<Token> it = sent.getTokens().iterator();
    BIO2Tag current = BIO2Tag.O(), last;
    BIOESTag tag;
    while(it.hasNext()) {
      t = it.next();
      current = t.getTag(source, BIO2Tag.class);
      last = l.getTag(source, BIO2Tag.class);
      if(last.isB() && current.isB()) tag = BIOESTag.S();
      else if(last.isB() && current.isI()) tag = BIOESTag.B();
      else if(last.isB() && current.isO()) tag =  BIOESTag.S();
      else if(last.isI() && current.isB()) tag = BIOESTag.E();
      else if(last.isI() && current.isI()) tag = BIOESTag.I();
      else if(last.isI() && current.isO()) tag = BIOESTag.E();
      else tag = BIOESTag.O();
      l.putTag(source, tag.setConfidence(last.getConfidence()));
      l = t;
    }
    if(current.isB()) tag = BIOESTag.S();
    else if(current.isI()) tag = BIOESTag.E();
    else tag = BIOESTag.O();
    t.putTag(source, tag.setConfidence(current.getConfidence()));
  }
  
}
