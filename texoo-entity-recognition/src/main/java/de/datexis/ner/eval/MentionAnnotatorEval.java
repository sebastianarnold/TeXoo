package de.datexis.ner.eval;

import com.google.common.collect.Lists;
import de.datexis.evaluation.ModelEvaluation;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Token;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.ner.MentionAnnotation;
import org.nd4j.common.primitives.Counter;

import java.util.List;
import java.util.TreeMap;
import java.util.stream.Collectors;

import static de.datexis.evaluation.ModelEvaluation.Measure.*;

/**
 * Evaluates Precision/Recall/F1 for span-based annotation (e.g. NER)
 * @author sarnold
 */
@Deprecated
public class MentionAnnotatorEval extends ModelEvaluation {
  
  Annotation.Source expectedSource;
  Annotation.Source predictedSource;
  
  public MentionAnnotatorEval(String experimentName) {
    this(experimentName, Annotation.Source.GOLD, Annotation.Source.PRED);
  }
  
  public MentionAnnotatorEval(String experimentName, Annotation.Source expected, Annotation.Source predicted) {
    super(experimentName);
    this.expectedSource = expected;
    this.predictedSource = predicted;
  }

  // please set train and test after training!
  @Deprecated
  public MentionAnnotatorEval(String experimentName, Dataset train, Dataset test) {
    super(experimentName, train, test);
  }
  
  public void clear() {
    counts = new TreeMap<>();
    counts.put(TP, new Counter<>());
    counts.put(FP, new Counter<>());
    counts.put(TN, new Counter<>());
    counts.put(FN, new Counter<>());
  }
  
  public void evaluateAnnotations() {
    int i = 0;
    for(Document d : test.getDocuments()) {
      counts.get(TP).setCount(i, getTP(d));
      counts.get(FP).setCount(i, getFP(d));
      counts.get(TN).setCount(i, getTN(d)); 
      counts.get(FN).setCount(i, getFN(d));
      i++;
    }
    // FIXME: required to update totalCount() - fixed in next Nd4j https://github.com/deeplearning4j/nd4j/commit/2698b2e23d8ccf6cf71c3bf6fc325e9638877ae8
    counts.get(TP).removeKey(-1);
    counts.get(FP).removeKey(-1);
    counts.get(TN).removeKey(-1);
    counts.get(FN).removeKey(-1);
  }
  
  public double getTP() {
    return counts.get(TP).totalCount();
  }
  
  public double getFP() {
    return counts.get(FP).totalCount();
  }
  
  public double getTN() {
    return counts.get(TN).totalCount();
  }
  
  public double getFN() {
    return counts.get(FN).totalCount();
  }
  
  private double getTP(Document d) {
    int result = 0;
    List<MentionAnnotation> predicted = Lists.newArrayList(d.streamAnnotations(predictedSource, MentionAnnotation.class).iterator());
    List<MentionAnnotation> expected = Lists.newArrayList(d.streamAnnotations(expectedSource, MentionAnnotation.class).iterator());
    for(MentionAnnotation pred : predicted) { 
      // was: if(expected.contains(pred)) result++;
      // TODO: optimize inner loops or use streams with match equality function
      for(MentionAnnotation exp : expected) {
        if(pred.matches(exp, Annotation.Match.STRONG)) {
          result++;
          break; // allow only one match
        }
      }
    }
    return result;
  }
  
  private double getFP(Document d) {
    int result = 0;
    List<MentionAnnotation> predicted = Lists.newArrayList(d.streamAnnotations(predictedSource, MentionAnnotation.class).iterator());
    List<MentionAnnotation> expected = Lists.newArrayList(d.streamAnnotations(expectedSource, MentionAnnotation.class).iterator());
    for(MentionAnnotation pred : predicted) {
      // was: if(!expected.contains(pred)) result++;
      boolean found = false;
      for(MentionAnnotation exp : expected) {
        if(exp.matches(pred, Annotation.Match.STRONG)) {
          found = true;
          break;
        }
      }
      if(!found) result++;
    }
    return result;
  }
  
  private double getTN(Document d) {
    // no annotation is explicitly NOT in test
    return 0;
  }
  
  private double getFN(Document d) {
    int result = 0;
    List<MentionAnnotation> predicted = Lists.newArrayList(d.streamAnnotations(predictedSource, MentionAnnotation.class).iterator());
    List<MentionAnnotation> expected = Lists.newArrayList(d.streamAnnotations(expectedSource, MentionAnnotation.class).iterator());
    for(MentionAnnotation exp : expected) {
      // was: if(!predicted.contains(exp)) result++;
      boolean found = false;
      for(MentionAnnotation pred : predicted) {
        if(pred.matches(exp, Annotation.Match.STRONG)) {
          found = true;
          break;
        }
      }
      if(!found) result++;
    }
    return result;
  }
  
  public double precision() {
    return getMicroPrecision(test);
  }
  
  /**
   * This is the CoNLL2003 Precision
   * @param data
   * @return precision = correctChunk / foundGuessed
   */
  private double getMicroPrecision(Dataset data) {
    double correct = 0.; // TP
    double foundGuessed = 0.; // TP + FP
    for(Document d : data.getDocuments()) {
      correct += getTP(d);
    }
    for(Document d : data.getDocuments()) {
      foundGuessed += getTP(d) + getFP(d);
    }
    if(foundGuessed > 0) return correct / foundGuessed;
    else return 0;
  }

  private double getMacroPrecision(Dataset data) {
    double prec = 0.;
    for(Document d : data.getDocuments()) {
      prec += getTP(d) / (getTP(d) + getFP(d)); //d.getAnnotations(test).size();
    }
    return prec / data.countDocuments();
  }
  
  public double recall() {
    return getMicroRecall(test);
  }
  
  /**
   * This is the CoNLL2003 Recall
   * @param data
   * @return recall = correctChunk / foundCorrect
   */
  public double getMicroRecall(Dataset data) {
    double correct = 0.; // TP
    double foundCorrect = 0.; // TP + FN
    for(Document d : data.getDocuments()) {
      correct += getTP(d);
    }
    for(Document d : data.getDocuments()) {
      foundCorrect += getTP(d) + getFN(d);
    }
    if(foundCorrect > 0) return correct / foundCorrect;
    else return 0;
  }

  private double getMacroRecall(Dataset data) {
    double prec = 0.;
    for(Document d : data.getDocuments()) {
      prec += getTP(d) / (getTP(d) + getFN(d));
    }
    return prec / data.countDocuments();
  }
  
  /**
   * This is CoNLL2003 Accuracy
   * @param data
   * @return accuracy = correctTags / tokenCounter
   */
  public double getTAccuracy(Dataset data) {
    double count = 0, correct = 0;
    for(Token t: data.streamTokens().collect(Collectors.toList())) {
      if(t.getTag(expectedSource, BIO2Tag.class).get().equals(t.getTag(predictedSource, BIO2Tag.class).get())) correct++;
      count++;
    }
    return correct / count;
  }
  
  public double f1() {
    return getMicroF1(test);
  }
  
  /**
   * This is CoNLL2003 NER-style F1
   * $FB1 = 2*$precision*$recall/($precision+$recall) if ($precision+$recall > 0);
   * @param data
   * @return 
   */
  public double getMicroF1(Dataset data) {
    return (2. * getMicroPrecision(data) * getMicroRecall(data))
         / (getMicroPrecision(data) + getMicroRecall(data));
  }

  private double getMacroF1(Dataset data) {
    return (2. * getMacroPrecision(data) * getMacroRecall(data))
         / (getMacroPrecision(data) + getMacroRecall(data));
  }
   
  public String printAnnotationStats() {
    StringBuilder line = new StringBuilder();
    line.append("ANNOTATION [micro-avg]\n")
        .append("#Docs\t#Tokns\t#Anns\t#Pred\t#TP\t#FP\t#TN\t#FN\tTAcc\tPrec\tRec\tF1");
    line.append("\n");
    line.append(fInt(test.countDocuments())).append("\t");
    line.append(fInt(test.countTokens())).append("\t");
    line.append(fInt(test.countAnnotations(expectedSource))).append("\t");
    line.append(fInt(test.countAnnotations(predictedSource))).append("\t");
    line.append(fInt(counts.get(TP).totalCount())).append("\t");
    line.append(fInt(counts.get(FP).totalCount())).append("\t");
    line.append(fInt(counts.get(TN).totalCount())).append("\t");
    line.append(fInt(counts.get(FN).totalCount())).append("\t");
    line.append(fDbl(getTAccuracy(test))).append("\t");
    line.append(fDbl(getMicroPrecision(test))).append("\t");
    line.append(fDbl(getMicroRecall(test))).append("\t");
    line.append(fDbl(getMicroF1(test))).append("\t");
    line.append("\n");
    System.out.println(line.toString());
    return line.toString();
  }
  
}
