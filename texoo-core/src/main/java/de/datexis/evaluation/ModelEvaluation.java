package de.datexis.evaluation;

import de.datexis.common.Timer;
import static de.datexis.evaluation.ModelEvaluation.Measure.*;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import java.util.Locale;
import java.util.TreeMap;
import org.nd4j.linalg.primitives.Counter;

/**
 * Common methods for model evaluation
 * @author sarnold
 */
public class ModelEvaluation {
  
  protected String experimentName;
  protected String experimentId;
  protected Dataset train, test;
  
  public static enum Measure {TP, FP, TN, FN};
  
  protected TreeMap<Measure,Counter<Integer>> counts;
  
  protected long trainCount, trainTime;
  protected long testCount, testTime;
    
  // we don't need the datasets here any more! hooray.
  @Deprecated
  public ModelEvaluation(String experimentName, Dataset train, Dataset test) {
    this(experimentName);
    setTrainDataset(train, 0, 0);
    setTestDataset(test, 0, 0);
  }
  
  public ModelEvaluation(String experimentName) {
    this.experimentName = experimentName;
    // FIXME: we need a uid;
    this.experimentId = "1234";
    clear();
  }
  
  public void setExperimentName(String name) {
    this.experimentName = name;
  }
  
  public void setTrainDataset(Dataset train, long trainCount, long trainTime) {
    this.train = train;
    this.trainCount = trainCount;
    this.trainTime = trainTime;
  }
  
  public void setTestDataset(Dataset test, long testCount, long testTime) {
    this.test = test;
    this.testCount = testCount;
    this.testTime = testTime;
  }
  
  public void clear() {
    counts = new TreeMap<>();
    counts.put(TP, new Counter<>());
    counts.put(FP, new Counter<>());
    counts.put(TN, new Counter<>());
    counts.put(FN, new Counter<>());
    trainCount = 0;
    trainTime = 0;
    testCount = 0;
    testTime = 0;
            
  }
  
  @Deprecated
  public void startTrainTimer() {
  }
  
  @Deprecated
  public long stopTrainTimer(int count) {
    return 0;
  }
  
  @Deprecated
  public void startTestTimer() {
  }
  
  @Deprecated
  public void stopTestTimer(int count) {
  }

  protected double seqL(Measure m, int c) {
    return (double) counts.get(m).getCount(c);
  }
  
  protected double div(double n, double d) {
    if(d == 0.0) return 0.0;
    else return n / d;
  }
  
  public String printExperimentStats() {
    StringBuilder line = new StringBuilder();
    line.append("==============================================================================\n");
    line.append("EXPERIMENT:\t").append(experimentName).append("\n");
    line.append("==============================================================================\n");
    int i = 0;
    /*for(LayerConfiguration layer : layers) {
      line.append("LAYER ").append(i++).append(":\t").append(layer.getName()).append("\n");
      line.append(layer.getLayerStats());
      line.append("------------------------------------------------------------------------------\n");
    }*/
    System.out.println(line.toString());
    return line.toString();
  }
  
  public String printDatasetStats() {
    StringBuilder line = new StringBuilder();
    if(train != null) {
      line.append("TRAIN:\t").append(train.getName()).append("\n");
      line.append("#Docs\t#Sents\t#Tokens\t#Anns\t#Examples\tTime\n");
      line.append(String.format(Locale.ROOT, "%,d",train.countDocuments())).append("\t");
      line.append(String.format(Locale.ROOT, "%,d",train.countSentences())).append("\t");
      line.append(String.format(Locale.ROOT, "%,d",train.countTokens())).append("\t");
      line.append(String.format(Locale.ROOT, "%,d",train.countAnnotations(Annotation.Source.GOLD))).append("\t");
      line.append(String.format(Locale.ROOT, "%,d",trainCount)).append("\t");
      line.append(Timer.millisToLongDHMS(trainTime)).append("\n");
      line.append("------------------------------------------------------------------------------\n");
    }
    if(test != null) {
      line.append("TEST:\t").append(test.getName()).append("\n");
      line.append("#Docs\t#Sents\t#Tokens\t#Anns\t#Examples\tTime\n");
      line.append(String.format(Locale.ROOT, "%,d",test.countDocuments())).append("\t");
      line.append(String.format(Locale.ROOT, "%,d",test.countSentences())).append("\t");
      line.append(String.format(Locale.ROOT, "%,d",test.countTokens())).append("\t");
      line.append(String.format(Locale.ROOT, "%,d",test.countAnnotations(Annotation.Source.GOLD))).append("\t");
      line.append(String.format(Locale.ROOT, "%,d",testCount)).append("\t");
      line.append(Timer.millisToLongDHMS(testTime)).append("\n");
      line.append("------------------------------------------------------------------------------\n");
    }
    System.out.println(line.toString());
    return line.toString();
  }
  
  public String printAnnotationStats() {
    StringBuilder line = new StringBuilder();
    line.append("ANNOTATION [micro-avg]\n")
        .append("Conf\t#Docs\t#Anns\t#TP\t#FP\t#TN\t#FN\tAcc\tPrec\tRec\tF1");
    System.out.println(line.toString());
    return line.toString();
  }
  
  protected String fDbl(double d) {
    return String.format(Locale.ROOT, "%6.2f", d * 100);
  }
  
  protected String fInt(double d) {
    return String.format(Locale.ROOT, "%6d", (int)d);
  }
  
}
