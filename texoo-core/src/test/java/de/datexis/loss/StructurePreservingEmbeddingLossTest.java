package de.datexis.loss;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;
import static org.hamcrest.Matchers.*;

import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


public class StructurePreservingEmbeddingLossTest {


  /*
    Check if the correct score is computed for two 2D Embeddings with data points aligned like the following
    figure: 
    +3                           
   
    +2             1o            
   
    +1             2o            
   
    +0       1x 2x    3x 4x      
   
    -1             3o            
   
    -2             4o            
   
    -3                           
   
    -5 -4 -3 -2 -1 +0 +1 +2 +4 +5
   */
  @Test
  public void computeScore() {
    StructurePreservingEmbeddingLoss loss = new StructurePreservingEmbeddingLoss();
    INDArray preOutput = setUp2DTestCase();

    INDArray labels = Nd4j.create(preOutput.shape());
    INDArray mask = null;

    IActivation activation = new ActivationIdentity();
    boolean average = false;

    double actualScore = loss.computeScore(labels, preOutput, activation, mask, average);
    
    assertThat(actualScore, is(closeTo(7.54, 0.01)));
  }

  @NotNull
  private INDArray setUp2DTestCase() {
    INDArray preOutput = Nd4j.create(4, 4);
    preOutput.putRow(0, Nd4j.create(new float[]{0,2,-2,0}));
    preOutput.putRow(1, Nd4j.create(new float[]{0,1,-1,0}));
    preOutput.putRow(2, Nd4j.create(new float[]{0,-1,1,0}));
    preOutput.putRow(3, Nd4j.create(new float[]{0,-2,2,0}));
    return preOutput;
  }  
  
  @NotNull
  private INDArray setUp2DTestCaseWithPositiveEpsilonDiversion(float epsilon) {
    INDArray preOutput = Nd4j.create(4, 4);
    preOutput.putRow(0, Nd4j.create(new float[]{0,2,-2,0}).add(epsilon));
    preOutput.putRow(1, Nd4j.create(new float[]{0,1,-1,0}).add(epsilon));
    preOutput.putRow(2, Nd4j.create(new float[]{0,-1,1,0}));
    preOutput.putRow(3, Nd4j.create(new float[]{0,-2,2,0}));
    return preOutput;
  }  
  
  @NotNull
  private INDArray setUp2DTestCaseWithNegativeEpsilonDiversion(float epsilon) {
    INDArray preOutput = Nd4j.create(4, 4);
    preOutput.putRow(0, Nd4j.create(new float[]{0,2,-2,0}).sub(epsilon));
    preOutput.putRow(1, Nd4j.create(new float[]{0,1,-1,0}).sub(epsilon));
    preOutput.putRow(2, Nd4j.create(new float[]{0,-1,1,0}));
    preOutput.putRow(3, Nd4j.create(new float[]{0,-2,2,0}));
    return preOutput;
  }

  @Test
  public void computeScoreShouldReturnZeroForOptimalSituation() {
    StructurePreservingEmbeddingLoss loss = new StructurePreservingEmbeddingLoss();
    INDArray preOutput = Nd4j.create(4,4);
    preOutput.putRow(0, Nd4j.create(new float[]{0,2,0,2}));
    preOutput.putRow(1, Nd4j.create(new float[]{0,1,0,1}));
    preOutput.putRow(2, Nd4j.create(new float[]{1,0,1,0}));
    preOutput.putRow(3, Nd4j.create(new float[]{2,0,2,0}));

    INDArray labels = Nd4j.create(preOutput.shape());
    INDArray mask = null;

    IActivation activation = new ActivationIdentity();
    boolean average = false;

    double actualScore = loss.computeScore(labels, preOutput, activation, mask, average);
    
    assertThat(actualScore, is(closeTo(0, 0.01)));
  }


  /*
  Check if the correct score is computed for two 2D Embeddings with data points aligned like the following
  figure: 
  +3                           
 
  +2             1o            
 
  +1             2o            
 
  +0       1x 2x    3x 4x      
 
  -1             3o            
 
  -2             4o            
 
  -3                           
 
  -5 -4 -3 -2 -1 +0 +1 +2 +4 +5
 */
  @Test
  public void computeScoreArray() {
    StructurePreservingEmbeddingLoss loss = new StructurePreservingEmbeddingLoss();
    INDArray preOutPut = setUp2DTestCase();
    
    INDArray labels = Nd4j.create(preOutPut.shape());

    INDArray scoreArray = loss.computeScoreArray(labels, preOutPut, new ActivationIdentity(), null);
    
    INDArray expected = Nd4j.create(4,1);
    expected.put(0,0, 3.1847);
    expected.put(1,0, 2);
    expected.put(2,0, 0.3563);
    expected.put(3,0, 2);
    
    assertThat(scoreArray, is(equalTo(expected)));
  }

  @Test
  public void computeGradient() {
    StructurePreservingEmbeddingLoss loss = new StructurePreservingEmbeddingLoss();
    INDArray defaultTestCase = setUp2DTestCase();
    INDArray slightlyPositiveCase = setUp2DTestCaseWithPositiveEpsilonDiversion(0.01f);
    INDArray slightlyNegativeCase = setUp2DTestCaseWithNegativeEpsilonDiversion(0.001f);

    INDArray labels = Nd4j.create(defaultTestCase.shape());
    INDArray mask = null;

    IActivation activation = new ActivationIdentity();

    INDArray actualGradients = loss.computeGradient(labels, defaultTestCase, activation, mask);
    INDArray positiveScores = loss.computeScoreArray(labels, slightlyPositiveCase, activation, mask);
    INDArray negativeScores = loss.computeScoreArray(labels, slightlyNegativeCase, activation, mask);
    INDArray estimatedGradients = negativeScores.sub(positiveScores);
    
    actualGradients.equalsWithEps(estimatedGradients,0.001f);
  }  
  
  @Test
  public void backwardGraphShouldBeExecutable() {
    StructurePreservingEmbeddingLoss loss = new StructurePreservingEmbeddingLoss();
    INDArray defaultTestCase = setUp2DTestCase();

    INDArray labels = Nd4j.create(defaultTestCase.shape());
    INDArray mask = null;

    IActivation activation = new ActivationIdentity();

   loss.computeGradient(labels, defaultTestCase, activation, mask);
  }  
  
  @Test
  public void forwardGraphShouldBeExecutable() {
    StructurePreservingEmbeddingLoss loss = new StructurePreservingEmbeddingLoss();
    INDArray defaultTestCase = setUp2DTestCase();

    INDArray labels = Nd4j.create(defaultTestCase.shape());
    INDArray mask = null;

    IActivation activation = new ActivationIdentity();

    loss.computeScore(labels, defaultTestCase, activation, mask, false);
  }

  @Test
  public void computeGradientAndScore() {
  }
  
  @Test
  public void testEuclideanDistanceBackwardsPass (){
    INDArray defaultTestCase = setUp2DTestCase();
    long singleEmbeddingSize = defaultTestCase.size(1) / 2L;

    // Split vectors
    INDArray x = defaultTestCase.get(NDArrayIndex.all(), NDArrayIndex.interval(0, singleEmbeddingSize));
    INDArray y = defaultTestCase.get(NDArrayIndex.all(), NDArrayIndex.interval(singleEmbeddingSize, defaultTestCase.size(1)));
    SameDiff graph = SameDiff.create();

    SDVariable x1 = graph.var("x", x);
    SDVariable y1 = graph.var("y", y);
    SDVariable transpose = graph.transpose(x1);
    SDVariable transpose1 = graph.transpose(y1);
    SDVariable eclidean = graph.euclideanDistance(transpose, transpose1, 0);
       
    System.out.println(graph.summary());
    INDArray withTranspose = graph.execAndEndResult();
    
    SameDiff graph2 = SameDiff.create();
    SDVariable x2 = graph2.var("x2", x);
    SDVariable y2 = graph2.var("y2", y);
    graph2.euclideanDistance(x2, y2,1);
    INDArray withoutTranspose = graph2.execAndEndResult();
    System.out.println(graph2.summary());

    assertThat(withTranspose, is(equalTo(withoutTranspose)));

    graph.execBackwardAndEndResult();
  }
}