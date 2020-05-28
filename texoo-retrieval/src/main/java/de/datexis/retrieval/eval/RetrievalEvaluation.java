package de.datexis.retrieval.eval;

import de.datexis.annotator.AnnotatorEvaluation;
import de.datexis.model.*;
import de.datexis.retrieval.model.ScoredResult;
import org.nd4j.common.util.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class RetrievalEvaluation extends AnnotatorEvaluation {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected double mrrsum = 0., mapsum = 0., recallNsum = 0.;
  protected double[] precisionKsum = new double[] {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  protected double[] recallKsum = new double[] {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  protected double[] dcgKsum = new double[] {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  protected double[] idcgKsum = new double[] {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  protected double[] ndcgKsum = new double[] {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  
  public RetrievalEvaluation(String experimentName) {
    super(experimentName,Annotation.Source.GOLD, Annotation.Source.PRED);
  }
  
  public void evaluateQueries(Dataset corpus) {
    evaluateQueries(corpus.getQueries());
  }
  
  public void evaluateQueries(Collection<Query> queries) {
    for(Query q : queries) {
      
      // expected results (might be relevant or non-relevant) in relevance order
      Collection<Result> expected = q.getResults(Annotation.Source.GOLD, Result.class);
      // predicted results in score order
      List<ScoredResult> predicted = q.getResults(Annotation.Source.PRED, ScoredResult.class);
      double[] idcg = new double[] {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  
      // assign ranks to predictions and initialize them as not relevant
      AtomicInteger rank = new AtomicInteger(0);
      predicted.stream().forEach(pred -> {
        pred.setRank(rank.incrementAndGet());
        pred.setRelevance(0);
      });
      
      // match all expected results to assign IDCG, relevance and scores
      int p = 0;
      double idcgSum = 0;
      for(Result exp : expected) {
        p++; // ideal position
        if(p <= 10) {
          // IDCG
          idcgSum += getDCGlog(exp.getRelevance(), p);
          idcg[p] = idcgSum;
          idcgKsum[p] += idcgSum;
        }
        predicted.stream().forEach(pred -> {
          if(pred.matches(exp)) {
            pred.setRelevant(exp.isRelevant());
            pred.setRelevance(exp.getRelevance());
          }
        });
      }
      // fill values when retrieved < p
      while(p < 10) {
        p++;
        idcg[p] = idcgSum;
        idcgKsum[p] += idcgSum;
      }
      
      // MRR
      Optional<? extends Result> first = predicted.stream()
        .filter(pred -> pred.isRelevant())
        .findFirst();
      if(first.isPresent()) mrrsum += (1. / first.get().getRank());
      
      int relevantPred = 0; // number of relevant documents among the retrieved ones
      double averagePrec = 0;
      double dcgSum = 0;
      long relevantExp = expected.stream()
        .filter(exp -> exp.isRelevant())
        .count();
      int k = 0;
      for(Result pred : predicted) {
        k++;
        assert k == pred.getRank(); // assumes that rs are sorted
        if(pred.isRelevant()) relevantPred++;
        if(k <= 10) {
          // P@k / R@k
          precisionKsum[k] += div(relevantPred, k);
          recallKsum[k]  += div(relevantPred, relevantExp);
          // DCG
          dcgSum += getDCGlog(pred.getRelevance(), k);
          dcgKsum[k] += dcgSum;
          // nDCG
          ndcgKsum[k] += dcgSum / idcg[k];
        }
        if(pred.isRelevant()) averagePrec += div(relevantPred, k);
        if(relevantPred >= relevantExp) break; // we found all so we can stop
      }
      // fill values when retrieved < k
      while(k < 10) {
        k++;
        precisionKsum[k] += div(relevantPred, k);
        recallKsum[k]  += div(relevantPred, relevantExp);
        dcgKsum[k] += dcgSum; // unchanged when no more relevant documents appear
        ndcgKsum[k] += dcgSum / idcg[k];
      }
      // calculate recall@N as recall over all candidates
      recallNsum  += div(relevantPred, relevantExp);
      
      // MAP
      averagePrec = div(averagePrec, relevantExp);
      mapsum += averagePrec;
      
      countExamples++;
      
    }
  
    log.info("{} queries, {} examples MRR={} P@1={} P@3={} P@5={} R@1={} R@3={} MAP={}",
      queries.size(), countExamples,
      getMRR(), getPrecisionK(1), getPrecisionK(3), getPrecisionK(5),
      getRecallK(1), getRecallK(3), getMAP()
    );
  
  }
  
  protected double getDCGlog(int relevance, int p) {
    // Stanford / Kaggle definition
    return (MathUtils.pow(2, relevance) - 1) / MathUtils.log2(p + 1);
    // Standard definition
    //else return ((double) relevance) / MathUtils.log2(p + 1);
  }
  
  /**
   * @return Mean Reciprocal Rank (macro-averaged over all Queries)
   * https://en.wikipedia.org/wiki/Mean_reciprocal_rank
   */
  protected double getMRR() {
    return mrrsum / countExamples;
  }
  
  /**
   * @return Mean Average Precision
   * https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
   */
  public double getMAP() {
    return mapsum / countExamples;
  }
  
  /**
   * @return Precision@K (macro-averaged over all Queries)
   * https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K
   */
  public double getPrecisionK(int k) {
    if(k <= 0 || k > precisionKsum.length - 1)
      throw new IllegalArgumentException("illegal argument 0 < k <= 10");
    return precisionKsum[k] / countExamples;
  }
  
  /**
   * @return Recall@K (macro-averaged over all Queries)
   */
  public double getRecallK(int k) {
    if(k <= 0 || k > precisionKsum.length - 1)
      throw new IllegalArgumentException("illegal argument 0 < k <= 10");
    return recallKsum[k] / countExamples;
  }
  
  /**
   * @return Recall@N, the recall over all predictions (macro-averaged over all Queries)
   */
  public double getRecallN() {
    return recallNsum / countExamples;
  }
  
  /**
   * @return Discounted Cumulative Gain (macro-averaged over all Queries)
   * https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain
   */
  public double getDCG(int k) {
    if(k <= 0 || k > precisionKsum.length - 1)
      throw new IllegalArgumentException("illegal argument 0 < k <= 10");
    return dcgKsum[k] / countExamples;
  }
  
  /**
   * @return ideal DCG score (macro-averaged over all Queries)
   * However, this function makes no sense for examples > 1, because IDCG is used to normalize a DCG score per example!
   */
  protected double getIDCG(int k) {
    if(k <= 0 || k > precisionKsum.length - 1)
      throw new IllegalArgumentException("illegal argument 0 < k <= 10");
    return idcgKsum[k] / countExamples;
  }
  
  /**
   * @return normalized DCG score (nDCG, macro-averaged over all Queries)
   * https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
   */
  public double getNDCG(int k) {
    if(k <= 0 || k > precisionKsum.length - 1)
      throw new IllegalArgumentException("illegal argument 0 < k <= 10");
    return ndcgKsum[k] / countExamples;
  
  }
  
  /**
   * @return MAP Score
   */
  @Override
  public double getScore() {
    return getMAP();
  }
  
  @Override
  public void calculateScores(Collection<Document> docs) {
    throw new UnsupportedOperationException("RetrievalEvaluation requires a Dataset with Queries");
  }
  
  @Override
  public String printEvaluationStats() {
  
    StringBuilder line = new StringBuilder("\n");
    line.append("RETRIEVAL EVALUATION [macro-avg]\n");
    line.append("|queries|\t P@1\t P@3\t P@5\t P@10\t R@1\t R@3\t R@5\t R@10\t R@N\tnDCG@1\tnDCG@3\tnDCG@5\tnDCG@10\t MRR\t MAP\t");
    line.append("\n");
  
    // statistics
    line.append(fInt(this.countExamples())).append("\t");
  
    // Classification: label(s) per sentence
    line.append(fDbl(getPrecisionK(1))).append("\t");
    line.append(fDbl(getPrecisionK(3))).append("\t");
    line.append(fDbl(getPrecisionK(5))).append("\t");
    line.append(fDbl(getPrecisionK(10))).append("\t");
    line.append(fDbl(getRecallK(1))).append("\t");
    line.append(fDbl(getRecallK(3))).append("\t");
    line.append(fDbl(getRecallK(5))).append("\t");
    line.append(fDbl(getRecallK(10))).append("\t");
    line.append(fDbl(getRecallN())).append("\t");
    line.append(fDbl(getNDCG(1))).append("\t");
    line.append(fDbl(getNDCG(3))).append("\t");
    line.append(fDbl(getNDCG(5))).append("\t");
    line.append(fDbl(getNDCG(10))).append("\t");
    line.append(fDbl(getMRR())).append("\t");
    line.append(fDbl(getMAP())).append("\t");
    line.append("\n");
    System.out.println(line.toString());
    return line.toString();
    
  }
}
