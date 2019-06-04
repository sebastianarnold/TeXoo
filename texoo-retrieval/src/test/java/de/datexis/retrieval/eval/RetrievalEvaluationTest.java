package de.datexis.retrieval.eval;

import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Query;
import de.datexis.model.Result;
import de.datexis.model.impl.RelevanceResult;
import de.datexis.model.impl.ScoredResult;
import de.datexis.preprocess.DocumentFactory;
import org.hamcrest.collection.IsIn;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class RetrievalEvaluationTest {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected Query q = new Query();
  protected Query q1 = Query.create("cat");
  protected Query q2 = Query.create("tori");
  protected Query q3 = Query.create("virus");
  
  Document d1 = DocumentFactory.fromText("catten");
  Document d2 = DocumentFactory.fromText("cati");
  Document d3 = DocumentFactory.fromText("cats");
  Document d4 = DocumentFactory.fromText("torii");
  Document d5 = DocumentFactory.fromText("tori");
  Document d6 = DocumentFactory.fromText("toruses");
  Document d7 = DocumentFactory.fromText("viruses");
  Document d8 = DocumentFactory.fromText("virii");
  Document d9 = DocumentFactory.fromText("viri");
  
  /**
   * Initialize test sets from Wikipedia examples
   */
  public RetrievalEvaluationTest() {
    
    // user-provided relevance scores and predicted results
    // as in https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    q.addResult(new RelevanceResult(Annotation.Source.GOLD, d1, 3));
    q.addResult(new RelevanceResult(Annotation.Source.GOLD, d2, 2));
    q.addResult(new RelevanceResult(Annotation.Source.GOLD, d3, 3));
    q.addResult(new RelevanceResult(Annotation.Source.GOLD, d4, 0));
    q.addResult(new RelevanceResult(Annotation.Source.GOLD, d5, 1));
    q.addResult(new RelevanceResult(Annotation.Source.GOLD, d6, 2));
    q.addResult(new RelevanceResult(Annotation.Source.GOLD, d7, 3));
    q.addResult(new RelevanceResult(Annotation.Source.GOLD, d8, 2));
    // scored results are not added in order to be able to test sorting
    q.addResult(new ScoredResult(Annotation.Source.PRED, d2, 0.7));
    q.addResult(new ScoredResult(Annotation.Source.PRED, d1, 0.9));
    q.addResult(new ScoredResult(Annotation.Source.PRED, d4, 0.3));
    q.addResult(new ScoredResult(Annotation.Source.PRED, d6, 0.1));
    q.addResult(new ScoredResult(Annotation.Source.PRED, d3, 0.5));
    q.addResult(new ScoredResult(Annotation.Source.PRED, d5, 0.2));
  
    // user-provided relevance scores and predicted results
    // as in https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    q1.addResult(new RelevanceResult(Annotation.Source.GOLD, d1, 0));
    q1.addResult(new RelevanceResult(Annotation.Source.GOLD, d2, 0));
    q1.addResult(new RelevanceResult(Annotation.Source.GOLD, d3, 1));
    q2.addResult(new RelevanceResult(Annotation.Source.GOLD, d5, 1)); // we don't add 0 relevance results here
    q3.addResult(new RelevanceResult(Annotation.Source.GOLD, d7, 1)); // we don't add 0 relevance results here
    q1.addResult(new ScoredResult(Annotation.Source.PRED, d1, 0.9));
    q1.addResult(new ScoredResult(Annotation.Source.PRED, d2, 0.5));
    q1.addResult(new ScoredResult(Annotation.Source.PRED, d3, 0.1));
    q2.addResult(new ScoredResult(Annotation.Source.PRED, d4, 0.9));
    q2.addResult(new ScoredResult(Annotation.Source.PRED, d5, 0.5));
    q2.addResult(new ScoredResult(Annotation.Source.PRED, d6, 0.1));
    q3.addResult(new ScoredResult(Annotation.Source.PRED, d7, 0.9));
    q3.addResult(new ScoredResult(Annotation.Source.PRED, d8, 0.5));
    q3.addResult(new ScoredResult(Annotation.Source.PRED, d9, 0.1));
    
  }
  
  @Test
  public void testSorting() {
    List<? extends Result> results = q.getResults(Annotation.Source.GOLD);
    Assert.assertThat(results.get(0).getDocumentRef(), IsIn.oneOf(d1, d3, d7)); // relevance 3
    Assert.assertThat(results.get(1).getDocumentRef(), IsIn.oneOf(d1, d3, d7));
    Assert.assertThat(results.get(2).getDocumentRef(), IsIn.oneOf(d1, d3, d7));
    Assert.assertThat(results.get(3).getDocumentRef(), IsIn.oneOf(d2, d6, d8)); // relevance 2
    Assert.assertThat(results.get(4).getDocumentRef(), IsIn.oneOf(d2, d6, d8));
    Assert.assertThat(results.get(5).getDocumentRef(), IsIn.oneOf(d2, d6, d8));
    Assert.assertThat(results.get(6).getDocumentRef(), IsIn.oneOf(d5)); // relevance 1
    Assert.assertThat(results.get(7).getDocumentRef(), IsIn.oneOf(d4)); // relevance 0
    Assert.assertEquals(8, results.size());
    results = q.getResults(Annotation.Source.PRED);
    Assert.assertEquals(results.get(0).getDocumentRef(), d1);
    Assert.assertEquals(results.get(1).getDocumentRef(), d2);
    Assert.assertEquals(results.get(2).getDocumentRef(), d3);
    Assert.assertEquals(results.get(3).getDocumentRef(), d4);
    Assert.assertEquals(results.get(4).getDocumentRef(), d5);
    Assert.assertEquals(results.get(5).getDocumentRef(), d6);
    Assert.assertEquals(6, results.size());
  }
  
  @Test
  public void testMRR() {
    
    RetrievalEvaluation eval;
    
    // as in https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q1));
    Assert.assertEquals(1, (int) eval.countExamples());
    assertEquals(1./3., eval.getMRR(), 0.001);
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q2));
    Assert.assertEquals(1, (int) eval.countExamples());
    assertEquals(1./2., eval.getMRR(), 0.001);
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q3));
    Assert.assertEquals(1, (int) eval.countExamples());
    assertEquals(1., eval.getMRR(), 0.001);
    
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Arrays.asList(q1, q2, q3));
    Assert.assertEquals(3, (int) eval.countExamples());
    assertEquals(11./18., eval.getMRR(), 0.001);
    
    // as in https://en.wikipedia.org/wiki/Discounted_cumulative_gain
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q));
    Assert.assertEquals(1, (int) eval.countExamples());
    assertEquals(1., eval.getMRR(), 0.001);
    
  }
  
  @Test
  public void testPrecisionRecallAtK() {
  
    RetrievalEvaluation eval;
  
    // as in https://en.wikipedia.org/wiki/Mean_reciprocal_rank
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q1));
    Assert.assertEquals(1, (int) eval.countExamples());
    assertEquals(0., eval.getPrecisionK(1), 0.001);
    assertEquals(0. / 2., eval.getPrecisionK(2), 0.001);
    assertEquals(1. / 3., eval.getPrecisionK(3), 0.001);
    assertEquals(0. / 1., eval.getRecallK(1), 0.001);
    assertEquals(0. / 1., eval.getRecallK(2), 0.001);
    assertEquals(1. / 1., eval.getRecallK(3), 0.001);
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q2));
    Assert.assertEquals(1, (int) eval.countExamples());
    assertEquals(0., eval.getPrecisionK(1), 0.001);
    assertEquals(1. / 2., eval.getPrecisionK(2), 0.001);
    assertEquals(1. / 3., eval.getPrecisionK(3), 0.001);
    assertEquals(0. / 1., eval.getRecallK(1), 0.001);
    assertEquals(1. / 1., eval.getRecallK(2), 0.001);
    assertEquals(1. / 1., eval.getRecallK(3), 0.001);
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q3));
    Assert.assertEquals(1, (int) eval.countExamples());
    assertEquals(1., eval.getPrecisionK(1), 0.001);
    assertEquals(1. / 2., eval.getPrecisionK(2), 0.001);
    assertEquals(1. / 3., eval.getPrecisionK(3), 0.001);
    assertEquals(1. / 1., eval.getRecallK(1), 0.001);
    assertEquals(1. / 1., eval.getRecallK(2), 0.001);
    assertEquals(1. / 1., eval.getRecallK(3), 0.001);
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Arrays.asList(q1, q2, q3));
    Assert.assertEquals(3, (int) eval.countExamples());
    assertEquals((0. + 0. + 1.) / 3., eval.getPrecisionK(1), 0.001);
    assertEquals((0. + 0.5 + 0.5) / 3., eval.getPrecisionK(2), 0.001);
    assertEquals((1.) / 3., eval.getPrecisionK(3), 0.001);
    assertEquals((0. + 0. + 1.) / 3., eval.getRecallK(1), 0.001);
    assertEquals((0. + 1. + 1.) / 3., eval.getRecallK(2), 0.001);
    assertEquals((1. + 1. + 1.) / 3., eval.getRecallK(3), 0.001);
  
    // as in https://en.wikipedia.org/wiki/Discounted_cumulative_gain
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q));
    Assert.assertEquals(1, (int) eval.countExamples());
    assertEquals(1., eval.getPrecisionK(1), 0.001);
    assertEquals(2. / 2., eval.getPrecisionK(2), 0.001);
    assertEquals(3. / 3., eval.getPrecisionK(3), 0.001);
    assertEquals(3. / 4., eval.getPrecisionK(4), 0.001);
    assertEquals(4. / 5., eval.getPrecisionK(5), 0.001);
    assertEquals(5. / 6., eval.getPrecisionK(6), 0.001);
    assertEquals(5. / 7., eval.getPrecisionK(7), 0.001);
    assertEquals(5. / 8., eval.getPrecisionK(8), 0.001);
    assertEquals(5. / 9., eval.getPrecisionK(9), 0.001);
    assertEquals(5. / 10., eval.getPrecisionK(10), 0.001);
    assertEquals(1. / 7., eval.getRecallK(1), 0.001);
    assertEquals(2. / 7., eval.getRecallK(2), 0.001);
    assertEquals(3. / 7., eval.getRecallK(3), 0.001);
    assertEquals(3. / 7., eval.getRecallK(4), 0.001);
    assertEquals(4. / 7., eval.getRecallK(5), 0.001);
    assertEquals(5. / 7., eval.getRecallK(6), 0.001);
    assertEquals(5. / 7., eval.getRecallK(7), 0.001);
    assertEquals(5. / 7., eval.getRecallK(8), 0.001);
    assertEquals(5. / 7., eval.getRecallK(9), 0.001);
    assertEquals(5. / 7., eval.getRecallK(10), 0.001);
  
  }
  
  @Test
  public void testMeanAveragePrecision() {
  
    RetrievalEvaluation eval;
    
    // as in https://en.wikipedia.org/wiki/Mean_reciprocal_rank
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q1));
    assertEquals(1. / 3., eval.getMAP(), 0.001);
    assertEquals(eval.getPrecisionK(3), eval.getMAP(), 0.001);
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q2));
    assertEquals(1. / 2., eval.getMAP(), 0.001);
    assertEquals(eval.getPrecisionK(2), eval.getMAP(), 0.001);
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q3));
    assertEquals(1. / 1., eval.getMAP(), 0.001);
    assertEquals(eval.getPrecisionK(1), eval.getMAP(), 0.001);
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Arrays.asList(q1, q2, q3));
    Assert.assertEquals(3, (int) eval.countExamples());
    assertEquals((1./1. + 1./2. + 1./3.) / 3., eval.getMAP(), 0.001);
  
    // as in https://en.wikipedia.org/wiki/Discounted_cumulative_gain
  
    eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q));
    assertEquals((1. + 2./2. + 3./3. + 4./5. + 5./6.) / 7., eval.getMAP(), 0.001);
    assertEquals((eval.getPrecisionK(1) + eval.getPrecisionK(2) + eval.getPrecisionK(3) + eval.getPrecisionK(5) + eval.getPrecisionK(6)) / 7., eval.getMAP(), 0.001);
    
  }
  
  @Test
  public void testNDCG() {
    
    // as in https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    
    RetrievalEvaluation eval = new RetrievalEvaluation("test");
    eval.evaluateQueries(Collections.singleton(q));
    
    // original example
    //assertEquals(3./1. + 2./1.585 + 3./2. +  0./2.322 + 1./2.585 + 2./2.807, eval.getDCG(6));
    // but we are using the 2^reli - 1 variant
    assertEquals(7./1., eval.getDCG(1), 0.001);
    assertEquals(7./1. + 3./1.585 , eval.getDCG(2), 0.001);
    assertEquals(7./1. + 3./1.585 + 7./2., eval.getDCG(3), 0.001);
    assertEquals(7./1. + 3./1.585 + 7./2. +  0./2.322, eval.getDCG(4), 0.001);
    assertEquals(7./1. + 3./1.585 + 7./2. +  0./2.322 + 1./2.585, eval.getDCG(5), 0.001);
    assertEquals(7./1. + 3./1.585 + 7./2. +  0./2.322 + 1./2.585 + 3./2.807, eval.getDCG(6), 0.001);
    assertEquals(eval.getDCG(6), eval.getDCG(7), 0.001); // no more relevant results retrieved!
    assertEquals(eval.getDCG(6), eval.getDCG(8), 0.001);
    assertEquals(eval.getDCG(6), eval.getDCG(9), 0.001);
    assertEquals(eval.getDCG(6), eval.getDCG(10), 0.001);
  
    // ideal DCG for this query
    // TODO: this is per query, right? so we cannot get this publicly from the evalution object
    assertEquals(7./1., eval.getIDCG(1), 0.001);
    assertEquals(7./1. + 7./1.585 , eval.getIDCG(2), 0.001);
    assertEquals(7./1. + 7./1.585 + 7./2., eval.getIDCG(3), 0.001);
    assertEquals(7./1. + 7./1.585 + 7./2. +  3./2.322, eval.getIDCG(4), 0.001);
    assertEquals(7./1. + 7./1.585 + 7./2. +  3./2.322 + 3./2.585, eval.getIDCG(5), 0.001);
    assertEquals(7./1. + 7./1.585 + 7./2. +  3./2.322 + 3./2.585 + 3./2.807, eval.getIDCG(6), 0.001);
    assertEquals(7./1. + 7./1.585 + 7./2. +  3./2.322 + 3./2.585 + 3./2.807 + 1./3., eval.getIDCG(7), 0.001);
    assertEquals(7./1. + 7./1.585 + 7./2. +  3./2.322 + 3./2.585 + 3./2.807 + 1./3. + 0., eval.getIDCG(8), 0.001);
    assertEquals(7./1. + 7./1.585 + 7./2. +  3./2.322 + 3./2.585 + 3./2.807 + 1./3. + 0., eval.getIDCG(9), 0.001);
    assertEquals(7./1. + 7./1.585 + 7./2. +  3./2.322 + 3./2.585 + 3./2.807 + 1./3. + 0., eval.getIDCG(10), 0.001);
    
    assertEquals(eval.getDCG(1) / eval.getIDCG(1), eval.getNDCG(1), 0.001);
    assertEquals(eval.getDCG(2) / eval.getIDCG(2), eval.getNDCG(2), 0.001);
    assertEquals(eval.getDCG(3) / eval.getIDCG(3), eval.getNDCG(3), 0.001);
    assertEquals(eval.getDCG(4) / eval.getIDCG(4), eval.getNDCG(4), 0.001);
    assertEquals(eval.getDCG(5) / eval.getIDCG(5), eval.getNDCG(5), 0.001);
    assertEquals(eval.getDCG(6) / eval.getIDCG(6), eval.getNDCG(6), 0.001);
    assertEquals(eval.getDCG(7) / eval.getIDCG(7), eval.getNDCG(7), 0.001);
    assertEquals(eval.getDCG(8) / eval.getIDCG(8), eval.getNDCG(8), 0.001);
    assertEquals(eval.getDCG(9) / eval.getIDCG(9), eval.getNDCG(9), 0.001);
    assertEquals(eval.getDCG(10) / eval.getIDCG(10), eval.getNDCG(10), 0.001);
    
    assertEquals((7./1. + 3./1.585 + 7./2. +  0./2.322 + 1./2.585 + 3./2.807) /
      (7./1. + 7./1.585 + 7./2. +  3./2.322 + 3./2.585 + 3./2.807 + 1./3. + 0.), eval.getNDCG(10), 0.001);
  }
  
}
