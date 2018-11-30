package de.datexis.loss;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;
import static org.hamcrest.Matchers.*;

import org.junit.Test;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


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
    INDArray preOutput = Nd4j.create(4,4);
    preOutput.putRow(0, Nd4j.create(new float[]{0,2,-2,0}));
    preOutput.putRow(1, Nd4j.create(new float[]{0,1,-1,0}));
    preOutput.putRow(2, Nd4j.create(new float[]{0,-1,1,0}));
    preOutput.putRow(3, Nd4j.create(new float[]{0,-2,2,0}));

    INDArray labels = Nd4j.create(preOutput.shape());
    INDArray mask = null;

    IActivation activation = new ActivationIdentity();
    boolean average = false;

    double actualScore = loss.computeScore(labels, preOutput, activation, mask, average);
    
    assertThat(actualScore, is(closeTo(7.54, 0.01)));
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

  @Test
  public void computeScoreArray() {
  }

  @Test
  public void computeGradient() {
  }

  @Test
  public void computeGradientAndScore() {
  }
}