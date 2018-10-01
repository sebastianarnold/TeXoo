package de.datexis.ner.eval;

import com.google.common.collect.Lists;
import de.datexis.annotator.AnnotatorEvaluation;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import static de.datexis.annotator.AnnotatorEvaluation.Measure.*;
import de.datexis.model.Document;
import de.datexis.model.Token;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.ner.MentionAnnotation;
import java.util.Collection;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.Collectors;
import org.nd4j.linalg.primitives.Counter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Evaluates Precision/Recall/F1 for Annotation Matching.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MentionAnnotatorEvaluation extends AnnotatorEvaluation {

  protected static Logger log = LoggerFactory.getLogger(MentionAnnotatorEvaluation.class);
  protected TreeMap<Measure,Counter<Integer>> counts;
  
  Annotation.Match matchingStrategy;
  
  public MentionAnnotatorEvaluation(String experimentName, Annotation.Match matchingStrategy) {
    this(experimentName, Annotation.Source.GOLD, Annotation.Source.PRED, matchingStrategy);
  }
  
  public MentionAnnotatorEvaluation(String experimentName, Annotation.Source expected, Annotation.Source predicted, Annotation.Match matchingStrategy) {
    super(experimentName, expected, predicted);
    log = LoggerFactory.getLogger(MentionAnnotatorEvaluation.class);
    this.matchingStrategy = matchingStrategy;
    clear();
  }

  protected void clear() {
    counts = new TreeMap<>();
    counts.put(TP, new Counter<>());
    counts.put(FP, new Counter<>());
    counts.put(TN, new Counter<>());
    counts.put(FN, new Counter<>());
    countExamples = 0;
    countAnnotations = 0;
    countDocs = 0;
    countSentences = 0;
    countTokens = 0;
  }
  
  protected double getCount(Measure m, int classIdx) {
    return (double) counts.get(m).getCount(classIdx);
  }
  
  @Override
  public double getScore() {
    return getMicroF1();
  }
  
  @Override
  public void calculateScores(Dataset dataset) {
    calculateScoresFromAnnotations(dataset.getDocuments(), MentionAnnotation.class);
  }
  
  @Override
  public void calculateScores(Collection<Document> docs) {
    calculateScoresFromAnnotations(docs, MentionAnnotation.class);
  }
  
  public void calculateScoresFromAnnotations(Collection<Document> docs, Class<? extends Annotation> annotationClass) {
    int i = 0;
    for(Document d : docs) {
      counts.get(TP).setCount(i, getTP(d, annotationClass));
      counts.get(FP).setCount(i, getFP(d, annotationClass));
      counts.get(TN).setCount(i, getTN(d, annotationClass)); 
      counts.get(FN).setCount(i, getFN(d, annotationClass));
      countTokens += d.countTokens();
      countSentences += d.countSentences();
      countAnnotations += d.countAnnotations(expectedSource, annotationClass);
      countDocs++;
      i++;
    }
    fixCounters();
  }
  
  /**
   * required to update totalCount() - fixed in next Nd4j https://github.com/deeplearning4j/nd4j/commit/2698b2e23d8ccf6cf71c3bf6fc325e9638877ae8
   */
  protected void fixCounters() {
    counts.get(TP).removeKey(-1);
    counts.get(FP).removeKey(-1);
    counts.get(TN).removeKey(-1);
    counts.get(FN).removeKey(-1);
  }
  
  public double getTP(Document d, Class<? extends Annotation> annotationClass) {
    int result = 0;
    List<? extends Annotation> predicted = Lists.newArrayList(d.streamAnnotations(predictedSource, annotationClass).iterator());
    List<? extends Annotation> expected = Lists.newArrayList(d.streamAnnotations(expectedSource, annotationClass).iterator());
    for(Annotation pred : predicted) { 
      // was: if(expected.contains(pred)) result++;
      // TODO: optimize inner loops or use streams with match equality function
      for(Annotation exp : expected) {
        if(pred.matches(exp, matchingStrategy)) {
          result++;
          break; // allow only one match
        }
      }
    }
    countExamples += result;
    return result;
  }
  
  public double getFP(Document d, Class<? extends Annotation> annotationClass) {
    int result = 0;
    List<? extends Annotation> predicted = Lists.newArrayList(d.streamAnnotations(predictedSource, annotationClass).iterator());
    List<? extends Annotation> expected = Lists.newArrayList(d.streamAnnotations(expectedSource, annotationClass).iterator());
    for(Annotation pred : predicted) {
      // was: if(!expected.contains(pred)) result++;
      boolean found = false;
      for(Annotation exp : expected) {
        if(exp.matches(pred, matchingStrategy)) {
          found = true;
          break;
        }
      }
      if(!found) result++;
    }
    countExamples += result;
    return result;
  }
  
  public double getTN(Document d, Class<? extends Annotation> annotationClass) {
    // no annotation is explicitly NOT in test
    return 0;
  }
  
  public double getFN(Document d, Class<? extends Annotation> annotationClass) {
    int result = 0;
    List<? extends Annotation> predicted = Lists.newArrayList(d.streamAnnotations(predictedSource, annotationClass).iterator());
    List<? extends Annotation> expected = Lists.newArrayList(d.streamAnnotations(expectedSource, annotationClass).iterator());
    for(Annotation exp : expected) {
      // was: if(!predicted.contains(exp)) result++;
      boolean found = false;
      for(Annotation pred : predicted) {
        if(pred.matches(exp, Annotation.Match.STRONG)) {
          found = true;
          break;
        }
      }
      if(!found) result++;
    }
    return result;
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
  
  /** safe division, where n/0 = 0 */
  protected double div(double n, double d) {
    if(d == 0.0) return 0.0;
    else return n / d;
  }
  
  /**
   * This is CoNLL2003 Accuracy
   * @param data
   * @return accuracy = correctTags / tokenCounter
   */
  public double getTokenAccuracy(Dataset data) {
    double count = 0, correct = 0;
    for(Token t: data.streamTokens().collect(Collectors.toList())) {
      if(t.getTag(expectedSource, BIO2Tag.class).get().equals(t.getTag(predictedSource, BIO2Tag.class).get())) correct++;
      count++;
    }
    return correct / count;
  }
  
  /**
   * Micro/Macro Accuracy
   */
  public double getAccuracy() {
    double found = getTP();
    double correct = getTP() + getFN();
    if(correct > 0) return found / correct;
    else return 0;
  }
  
  /**
   * Accuracy per class
   * @param c - class index
   */
  protected double getAccuracy(int c) {
    return div(getCount(TP,c) , getCount(TP,c) + getCount(FN,c));
  }
  
  /**
   * Micro Precision (average precision over all examples).
   * This is the CoNLL2003 Precision.
   * @return precision = correctChunk / foundGuessed
   */
  public double getMicroPrecision() {
    double correct = getTP();
    double foundGuessed = getTP() + getFP();
    if(foundGuessed > 0) return correct / foundGuessed;
    else return 0;
  }
  
  
  /**
   * Macro Precision (average Precision over all classes).
   */
  public double getMacroPrecision() {
    double score = 0;
    int count = 0;
    for(int i = 0; i<countDocs; ++i) {
      if(getCount(FP,i) > 0) {
        score += getPrecision(i);
        count++;
      }
    }
    if(count > 0) return score / count;
    else return 0;
  }
  
  /**
   * Precision per class
   * @param c - class index
   */
  protected double getPrecision(int c) {
    if(getCount(TP,c) == 0) return 0;
    return div(getCount(TP,c) , getCount(TP,c) + getCount(FP,c));
  }
  
  /**
   * Micro Recall (average recall over all examples).
   * This is the CoNLL2003 Recall.
   * @return recall = correctChunk / foundCorrect
   */
  public double getMicroRecall() {
    double correct = getTP();
    double foundCorrect = getTP() + getFN();
    if(foundCorrect > 0) return correct / foundCorrect;
    else return 0;
  }
  
  /**
   * Macro Recall (average recall over all classes).
   */
  public double getMacroRecall() {
    double score = 0;
    int count = 0;
    for(int i = 0; i<countDocs; ++i) {
      if(getCount(FN,i) > 0) {
        score += getRecall(i);
        count++;
      }
    }
    if(count > 0) return score / count;
    else return 0;
  }
  
  /**
   * Recall per class
   * @param c - class index
   */
  protected double getRecall(int c) {
    if(getCount(TP,c) == 0) return 0;
    return div(getCount(TP,c) , getCount(TP,c) + getCount(FN,c));
  }
  
  /**
   * Micro F1 score (average F1 over all examples).
   * This is CoNLL2003 NER-style F1
   * @return $FB1 = 2*$precision*$recall/($precision+$recall) if ($precision+$recall > 0)
   */
  public double getMicroF1() {
    return getF1(getMicroPrecision(), getMicroRecall());
  }
  
  /**
   * Macro F1 score (average F1 over all classes).
   */
  public double getMacroF1() {
    return getF1(getMacroPrecision(), getMacroRecall());
  }
  
  /**
   * F1 score per document
   * @param i - document index
   */
  protected double getF1(int i) {
    return getF1(getPrecision(i), getRecall(i));
  }
  
  /**
   * F1 score from prec and recall
   */
  private double getF1(double precision, double recall) {
    if(precision + recall == 0) return 0;
    return (2. * precision * recall) / (precision + recall);
  }

  public String printAnnotationStats() {
    return printHeader() + printRow();
  }
  
  public static String printHeader() {
    StringBuilder line = new StringBuilder();
    line.append("ANNOTATION [micro-avg]\n")
        .append("Experiment ----------------------------------------\t#Docs\t#Tokns\t#Anns\t#Pred\t#TP\t#FP\t#TN\t#FN\tPrec\tRec\tF1");
    line.append("\n");
    System.out.print(line.toString());
    return line.toString();
  }
  
  public String printRow() {
    StringBuilder line = new StringBuilder();
    line.append(fStr(experimentName, 50)).append("\t");
    line.append(fInt(countDocuments())).append("\t");
    line.append(fInt(countTokens())).append("\t");
    line.append(fInt(countAnnotations())).append("\t");
    line.append(fInt(countExamples())).append("\t");
    line.append(fInt(getTP())).append("\t");
    line.append(fInt(getFP())).append("\t");
    line.append(fInt(getTN())).append("\t");
    line.append(fInt(getFN())).append("\t");
    line.append(fDbl(getMicroPrecision())).append("\t");
    line.append(fDbl(getMicroRecall())).append("\t");
    line.append(fDbl(getMicroF1())).append("\t");
    line.append("\n");
    System.out.print(line.toString());
    return line.toString();
  }

}
