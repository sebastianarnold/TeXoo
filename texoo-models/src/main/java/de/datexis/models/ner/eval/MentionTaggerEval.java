package de.datexis.models.ner.eval;

import de.datexis.evaluation.ModelEvaluation;
import static de.datexis.evaluation.ModelEvaluation.Measure.*;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Token;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.model.tag.BIOESTag;
import de.datexis.model.tag.Tag;
import java.util.ArrayList;
import java.util.stream.Collectors;
import org.deeplearning4j.eval.ConfusionMatrix;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Evaluates Precision/Recall/F1 for token-based labeling (e.g. BIOES)
 * @author sarnold
 */
public class MentionTaggerEval extends ModelEvaluation {
  
  protected int classes;
  protected Tag tagset;
  protected Evaluation eval;
  
  protected double accuracy;
  protected double precision;
  protected double recall;
  protected double f1;
  
  private ArrayList<Integer> examplesCurve;
  private ArrayList<Double> precisionCurve;
  private ArrayList<Double> recallCurve;
  private ArrayList<Double> f1Curve;
  private ArrayList<Double> errorCurve;
  
  Annotation.Source expectedSource;
  Annotation.Source predictedSource;
  
  
  public MentionTaggerEval(String experimentName) {
    this(experimentName, BIO2Tag.class);
  }
  
  public MentionTaggerEval(String experimentName, Class tagset) {
    this(experimentName, tagset, Annotation.Source.GOLD, Annotation.Source.PRED);
  }
  
  public MentionTaggerEval(String experimentName, Class tagset, Annotation.Source expected, Annotation.Source predicted) {
    super(experimentName);
    try {
      this.tagset = (Tag) tagset.newInstance();
    } catch (InstantiationException | IllegalAccessException ex) {
      //log.error("Could not instantiate target class " + target.getName());
    }
    this.classes = this.tagset.getVectorSize();
    this.expectedSource = expected;
    this.predictedSource = predicted;
  }

  // please set the test and train datasets after training!
  @Deprecated
  public MentionTaggerEval(String experimentName, Dataset train, Dataset test, Class tagset) {
    this(experimentName, tagset, Annotation.Source.GOLD, Annotation.Source.PRED);
    setTestDataset(test, 0, 0);
    setTrainDataset(train, 0, 0);
  }
  
  @Override
  public void clear() {
    super.clear();
    eval = new Evaluation(classes);
    examplesCurve = new ArrayList<>();
    precisionCurve = new ArrayList<>();
    recallCurve = new ArrayList<>();
    f1Curve = new ArrayList<>();
    errorCurve = new ArrayList<>();
  }
  
  public void eval(Token t, INDArray expected, INDArray predicted, boolean print) {
    eval.eval(expected, predicted);
    if(print) System.out.println(t.getText() + "\t" + expected + "\t" + predicted);
  }
  
  public void evalTimeSeries(INDArray expected, INDArray predicted) {
    eval.evalTimeSeries(expected, predicted);
  }
  
  public void evalTimeSeries(INDArray expected, INDArray predicted, INDArray labelsMask) {
    if(expected.shape()[2] == 1) eval.evalTimeSeries(expected.transpose(), predicted.transpose());
    else eval.evalTimeSeries(expected, predicted, labelsMask);
  }
  
  public void evalTimeSeries(Evaluation ev) {
    this.eval = ev;
  }
  
  public void appendTrainingCurve(double precision, double recall, double f1) {
    examplesCurve.add(0);
    precisionCurve.add(precision);
    recallCurve.add(recall);
    f1Curve.add(f1);
    errorCurve.add(0.);
  }
  
  public void appendTrainingCurve(int examples, double precision, double recall, double f1, double error) {
    examplesCurve.add(examples);
    precisionCurve.add(precision);
    recallCurve.add(recall);
    f1Curve.add(f1);
    errorCurve.add(error);
  }
    
  public void calculateMeasures(Dataset test) {
    for(int c = 0; c<classes; ++c) {
      double tp = 0, fp = 0, tn = 0, fn = 0;
      for(Token t : test.streamTokens().collect(Collectors.toList())) {
        String g = t.getTag(expectedSource, tagset.getClass()).getTag();
        String p = t.getTag(predictedSource, tagset.getClass()).getTag();
        String cl = tagset.getTag(c);
        if( g.equals(cl) &&  p.equals(cl)) tp++;
        if(!g.equals(cl) &&  p.equals(cl)) fp++;
        if(!g.equals(cl) && !p.equals(cl)) tn++;
        if( g.equals(cl) && !p.equals(cl)) fn++;
      }
      counts.get(TP).setCount(c, tp);
      counts.get(FP).setCount(c, fp);
      counts.get(TN).setCount(c, tn);
      counts.get(FN).setCount(c, fn);
    }
  }
  
  private void calculateMeasures(Evaluation eval) {
    ConfusionMatrix<Integer> m = eval.getConfusionMatrix();
    for(int c = 0; c<classes; ++c) {
      double tp = 0, fp = 0, tn =0, fn =0;
      for(int p = 0; p<classes; ++p) {
        for(int g = 0; g<classes; ++g) {
          int x = m.getCount(g,p);
          if(p==c && g==c) tp += x;
          if(p==c && g!=c) fp += x;
          if(p!=c && g!=c) tn += x;
          if(p!=c && g==c) fn += x;
        }
      }
      counts.get(TP).setCount(c, tp);
      counts.get(FP).setCount(c, fp);
      counts.get(TN).setCount(c, tn);
      counts.get(FN).setCount(c, fn);
    }
  }
  
  public String printSequenceStats() {
    StringBuilder line = new StringBuilder();
    line.append("SEQUENCE Training per Config [macro-avg]\t\t\t\tTrain Time [ms]\t\t\tTest Time [ms]\n")
        .append("Conf\t\t#EncMiss\t#TP\t#FP\t#TN\t#FN\tAcc\tPrec\tRec\tF1\t#Docs\t#Sents\t#Tokens\tTotal\tDoc\tSent\t#Docs\t#Sents\t#Tokens\tTotal\tDoc\tSent\n");
    System.out.println(line.toString());
    return line.toString();
  }
  
  public String printSequenceClassStats() {
    return printSequenceClassStats(true);
  }
  
  public String printSequenceClassStats(boolean calculate) {
    
    if(calculate) calculateMeasures(eval);
    
    StringBuilder line = new StringBuilder();
    line.append("SEQUENCE Labeling per Class [macro-avg]\n")
        .append("Class\t#Tokns\t#Enc\t    TP\t    FP\t    TN\t    FN\tAcc\tPrec\tRec\tF1\n");
    
    double acc = 0;
    double pre = 0;
    double rec = 0;
    for(int c = 0; c<classes; ++c) {
      line.append(tagset.getTag(c)).append("\t");
      line.append(fInt(counts.get(TP).getCount(c) + counts.get(FN).getCount(c))).append("\t");
      line.append("\t");
      line.append(fInt(counts.get(TP).getCount(c))).append("\t");
      line.append(fInt(counts.get(FP).getCount(c))).append("\t");
      line.append(fInt(counts.get(TN).getCount(c))).append("\t");
      line.append(fInt(counts.get(FN).getCount(c))).append("\t");
      line.append(fDbl(getAccuracy(c))).append("\t");
      line.append(fDbl(getPrecision(c))).append("\t");
      line.append(fDbl(getRecall(c))).append("\t");
      line.append(fDbl(getF1(c))).append("\t");
      line.append("\n");
      acc += getAccuracy(c);
      pre += getPrecision(c);
      rec += getRecall(c);
    }
    acc = acc/classes;
    pre = pre/classes;
    rec = rec/classes;
    line.append("Total\t");
    line.append(fInt(counts.get(TP).totalCount() + counts.get(FN).totalCount())).append("\t");
    line.append("\t");
    line.append(fInt(counts.get(TP).totalCount())).append("\t");
    line.append(fInt(counts.get(FP).totalCount())).append("\t");
    line.append(fInt(counts.get(TN).totalCount())).append("\t");
    line.append(fInt(counts.get(FN).totalCount())).append("\t");
    line.append(fDbl(acc)).append("\t");
    line.append(fDbl(pre)).append("\t");
    line.append(fDbl(rec)).append("\t");
    line.append(fDbl(getF1(pre,rec))).append("\t");
    line.append("\n");
    System.out.println(line.toString());
    return line.toString();
  }
  
  public String printTrainingCurve() {
    StringBuilder line = new StringBuilder();
    line.append("#\tCount\tPrec\tRec\tF1\tError\n");
    for(int i = 0; i<f1Curve.size(); ++i) {
      line.append(i).append("\t");
      line.append(fInt(examplesCurve.get(i))).append("\t");
      line.append(fDbl(precisionCurve.get(i))).append("\t");
      line.append(fDbl(recallCurve.get(i))).append("\t");
      line.append(fDbl(f1Curve.get(i))).append("\t");
      line.append(fDbl(errorCurve.get(i) / 100.));
      line.append("\n");
    }
    return line.toString();
  }
  
  private double getAccuracy(int c) {
    return div(seqL(TP,c) + seqL(TN,c) , seqL(TP,c) + seqL(TN,c) + seqL(FP,c) + seqL(FN,c));
  }
  
  private double getPrecision(int c) {
    return div(seqL(TP,c) , seqL(TP,c) + seqL(FP,c));
  }
  
  private double getRecall(int c) {
    return div(seqL(TP,c) , seqL(TP,c) + seqL(FN,c));
  }
  
  private double getF1(int c) {
    return getF1(getPrecision(c), getRecall(c));
  }
  
  private double getF1(double precision, double recall) {
    if(precision + recall == 0) return 0;
    return (2. * precision * recall) / (precision + recall);
  }
  
}
