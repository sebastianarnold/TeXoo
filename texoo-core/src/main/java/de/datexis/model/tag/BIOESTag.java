package de.datexis.model.tag;

import de.datexis.model.Annotation;
import de.datexis.model.Annotation.Source;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.SortedMap;
import java.util.TreeMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Tag that is used to label a single Token in BIOES format, used for sequence learning.
 * @author sarnold
 */
public class BIOESTag implements Tag {

  private static final Logger log = LoggerFactory.getLogger(BIOESTag.class);

  /**
   * The Label of the token in BIOES format, used for sequence learning
   * S = Single Token Annotation
   * B = Begin Annotation
   * I = Inside Annotation
   * E = End of Annotation
   * O = Outside Annotation
   */
  public static enum Label {
    S, B, I, E, O
  }
  
  public static Enum<?>[] getLabels() {
    return Label.values();
  }
    
  public static BIOESTag B() { return new BIOESTag(Label.B); }
  public static BIOESTag I() { return new BIOESTag(Label.I); }
  public static BIOESTag O() { return new BIOESTag(Label.O); }
  public static BIOESTag E() { return new BIOESTag(Label.E); }
  public static BIOESTag S() { return new BIOESTag(Label.S); }
    
  protected final Label label;
  protected final String type;
  protected final INDArray vector;
  protected double confidence = 0.;
  
  public BIOESTag() {
    this(Label.O, null);
  }
  
  public BIOESTag(Label tag, String type) {
    this.label = tag;
    this.type = type;
    this.vector = null;
    this.confidence = 1.;
  }
  
  public BIOESTag(INDArray predicted, String type, boolean storeVector) {
    this.label = max(predicted);
    this.type = type;
    this.vector = storeVector ? predicted.detach() : null;
    if(this.label.equals(Label.O)) this.confidence = predicted.getDouble(Label.O.ordinal());
    else this.confidence = 1. - predicted.getDouble(Label.O.ordinal());
  }
  
  public BIOESTag(Label tag) {
    this(tag, tag.equals(Label.O) ? null : GENERIC);
  }
  
  public BIOESTag(Label tag, INDArray predicted, boolean storeVector) {
    this.label = tag;
    this.type = tag.equals(Label.O) ? null : GENERIC;
    this.vector = storeVector ? predicted.detach() : null;
    this.confidence = predicted.getDouble(this.label.ordinal());
  }
  
  public BIOESTag(INDArray predicted, boolean StoreVector) {
    this(predicted, GENERIC, StoreVector);
  }
  
  public static String toString(INDArray vec) {
    return  String.format(Locale.ROOT, "%6.2f S\n%6.2f B\n%6.2f I\n%6.2f E\n%6.2f O",
        vec.getDouble(0), vec.getDouble(1), vec.getDouble(2), vec.getDouble(3), vec.getDouble(4));
  }

  public Label get() {
    return label;
  }
  
  public static INDArray getVector(Label tag) {
		switch(tag) {
		case S:
			return Nd4j.create(new double[] { 1, 0, 0, 0, 0 });
		case B:
			return Nd4j.create(new double[] { 0, 1, 0, 0, 0 });
    case I:
			return Nd4j.create(new double[] { 0, 0, 1, 0, 0 });
    case E:
			return Nd4j.create(new double[] { 0, 0, 0, 1, 0 });
    default:
    case O:
			return Nd4j.create(new double[] { 0, 0, 0, 0, 1 });
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
  
  public boolean isE() {
    return label.equals(Label.E);
  }
  
  public boolean isS() {
    return label.equals(Label.S);
  }
  
  
  @Override
  public double getConfidence() {
    return this.confidence;
  }
  
  public BIOESTag setConfidence(double confidence) {
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
    if(x == 0) return Label.S;
    else if(x == 1) return Label.B;
    else if(x == 2) return Label.I;
    else if(x == 3) return Label.E;
    else return Label.O;
  }
  
  /**
   * Returns TRUE, iff this sequence of labels is valid.
   * The Tokens are treated as complete sentence, i.e. sequence must begin 
   * and end correctly at sentence boundaries.
   * @return 
   */
  public static boolean isCorrect(Source source, Iterable<Token> tokens) {
    List<BIOESTag.Label> labels = new ArrayList<>();
    labels.add(Label.O);
    for(Token t : tokens) {
      labels.add(t.getTag(source, BIOESTag.class).get());
    }
    labels.add(Label.O);
    return isCorrect(labels.toArray(new BIOESTag.Label[0]));
  }
  
  /**
   * Returns TRUE, iff this sequence of labels is valid.
   * The label sequence is treated as a partition, i.e. no check for correct
   * begin and end at boundaries is done.
   * @param labels
   * @return 
   */
  private static boolean isCorrect(Label... labels) {
    if(labels.length == 0) return true;
    Label c = null;
    for(Label l : labels) {
      if(c == null) {
        c = l;
        continue;
      }
      if(c.equals(Label.S) && l.equals(Label.E)) return false;
      if(c.equals(Label.S) && l.equals(Label.I)) return false;
      if(c.equals(Label.B) && l.equals(Label.B)) return false;
      if(c.equals(Label.B) && l.equals(Label.O)) return false;
      if(c.equals(Label.B) && l.equals(Label.S)) return false;
      if(c.equals(Label.I) && l.equals(Label.B)) return false;
      if(c.equals(Label.I) && l.equals(Label.O)) return false;
      if(c.equals(Label.I) && l.equals(Label.S)) return false;
      if(c.equals(Label.E) && l.equals(Label.E)) return false;
      if(c.equals(Label.E) && l.equals(Label.I)) return false;
      if(c.equals(Label.O) && l.equals(Label.I)) return false;
      if(c.equals(Label.O) && l.equals(Label.E)) return false;
      c = l;
    }
    return true;
  }
  
  @Override
  public int getVectorSize() {
    return 5;
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
    final BIOESTag other = (BIOESTag) obj;
    if (!Objects.equals(this.type, other.type)) {
      return false;
    }
    if (this.label != other.label) {
      return false;
    }
    return true;
  }
  
  /**
   * Converts BIOES Tags to BIO2 Tags
   * @param source 
   */
  public static void convertToBIO2(Dataset data, Annotation.Source source) {
    for(Document doc : data.getDocuments()) {
      for(Sentence sent : doc.getSentences()) {
        convertToBIO2(sent, source);
      }
    }
  }
  
  /**
   * Converts BIOES Tags to BIO2 Tags
   * @param source 
   */
  public static void convertToBIO2(Document doc, Annotation.Source source) {
    for(Sentence sent : doc.getSentences()) {
      convertToBIO2(sent, source);
    }
  }
  
  /**
   * Converts BIOES Tags to BIO2 Tags
   * @param sent
   * @param source
   */
  public static synchronized void convertToBIO2(Sentence sent, Annotation.Source source) {
    Token t = new Token("");
    Iterator<Token> it = sent.getTokens().iterator();
    BIOESTag current;
    BIO2Tag tag;
    while(it.hasNext()) {
      t = it.next();
      current = t.getTag(source, BIOESTag.class);
      if(current.isB()) tag = BIO2Tag.B();
      else if(current.isS()) tag = BIO2Tag.B();
      else if(current.isI()) tag = BIO2Tag.I();
      else if(current.isE()) tag = BIO2Tag.I();
      else tag = BIO2Tag.O();
      tag.setConfidence(current.getConfidence());
      tag.setType(current.getType());
      t.putTag(source, tag);
    }
  }
  
  /**
   * Corrects the order of BIOES Tags similar to a CRF Classifier.
   */
  public static void correctCRF(Dataset data, Annotation.Source source) {
   for(Document doc : data.getDocuments()) {
      for(Sentence sent : doc.getSentences()) {
        correctCRF(sent, source);
      }
    }
  }
  
  /**
   * Corrects the order of BIOES Tags similar to a CRF Classifier.
   * @param doc
   * @param source 
   */
  public static synchronized void correctCRF(Sentence sent, Annotation.Source source) {

    List<Token> tokens;
    final SortedMap<Double,BIOESTag.Label[]> scores = new TreeMap<>();
    
    if(BIOESTag.isCorrect(source, sent.getTokens())) return;

    INDArray[] vec = new INDArray[5];
    BIOESTag.Label[] tag = new BIOESTag.Label[5];
    int size = 5;
    double score = 0;
    tokens = sent.getTokens();

    for(int cursor=0; cursor < tokens.size(); cursor++) {

      scores.clear();

      vec[0] = getLabelVector(tokens, cursor-1, source);
      vec[1] = getLabelVector(tokens, cursor, source);
      vec[2] = getLabelVector(tokens, cursor+1, source);
      vec[3] = getLabelVector(tokens, cursor+2, source);
      vec[4] = getLabelVector(tokens, cursor+3, source);

      // permutate
      for(int x=0; x<size; x++) {
       for(int y=0; y<size; y++) {
         for(int z=0; z<size; z++) {
           tag[0] = getLabel(tokens, cursor-1, source); // this is now fixed
           tag[1] = BIOESTag.index(x);
           tag[2] = BIOESTag.index(y);
           tag[3] = BIOESTag.index(z);
           tag[4] = BIOESTag.max(vec[4]);
           score = vec[1].getDouble(x) + vec[2].getDouble(y) + vec[3].getDouble(z); // use arithmetic mean, can leave out normalization
           if(BIOESTag.isCorrect(tag)) {
             //log.info("correct: " + Arrays.toString(l) + " (" + w[1].getDouble(x) + ", " + w[2].getDouble(y) + ", " + w[3].getDouble(z) + ")");
             //output += "\n   CORRECT: "  + Arrays.toString(l) + " (" + w[1].getDouble(x) + ", " + w[2].getDouble(y) + ", " + w[3].getDouble(z) + ")";
             scores.put(score, tag.clone());
           } else {
             //output += "\n incorrect: "  + Arrays.toString(l) + " (" + w[1].getDouble(x) + ", " + w[2].getDouble(y) + ", " + w[3].getDouble(z) + ")";
           }
         }
       }
      }

      try {
        tag = scores.get(scores.lastKey());
        //log.info("best: " + Arrays.toString(l));
        BIOESTag result = new BIOESTag(tag[1], vec[1], true);
        tokens.get(cursor).putTag(source, result);
      } catch(NoSuchElementException e) {
        log.warn("could not find correct labels for sentence '" + sent.toString() + "'");
        log.debug(scores.toString());
        log.debug(Arrays.deepToString(vec));
      }
    }
    
  }

  private static INDArray getLabelVector(List<Token> tokens, int pos, Annotation.Source source) {
    if(pos < 0 || pos >= tokens.size()) return BIOESTag.getVector(BIOESTag.Label.O);
    else return tokens.get(pos).getTag(source, BIOESTag.class).getVector();
  }
  
  private static Label getLabel(List<Token> tokens, int pos, Annotation.Source source) {
    if(pos < 0 || pos >= tokens.size()) return BIOESTag.Label.O;
    else return tokens.get(pos).getTag(source, BIOESTag.class).get();
  }
  
}
