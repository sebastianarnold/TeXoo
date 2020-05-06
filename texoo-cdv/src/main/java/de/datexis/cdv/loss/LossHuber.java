package de.datexis.cdv.loss;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.serde.jackson.shaded.NDArrayTextDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextSerializer;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * (Pseudo-) Huber loss function L = d^2 ( sqrt(1 + ((Ytrue-Ypred)/d)^2) - 1 )
 * https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class LossHuber implements ILossFunction {
  
  @JsonSerialize(using = NDArrayTextSerializer.class)
  @JsonDeserialize(using = NDArrayTextDeSerializer.class)
  protected final INDArray weights;
  
  protected final double DELTA = 2;
  
  public LossHuber() {
    this(null);
    
  }
  
  /**
   * L2 loss function where each the output is (optionally) weighted/scaled by a flags scalar value.
   * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
   * A weight vector of 1s should give identical results to no weight vector.
   *
   * @param weights Weights array (row vector). May be null.
   */
  public LossHuber(INDArray weights) {
    if (weights != null && !weights.isRowVector()) {
      throw new IllegalArgumentException("Weights array must be a row vector");
    }
    this.weights = weights;
  }
  
  protected INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    if(!labels.equalShapes(preOutput)){
      Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
    }
    labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
    INDArray output = activationFn.getActivation(preOutput.dup(), true);
    
    // Huber d^2 ( sqrt(1 + ((Ytrue-Ypred)/d)^2) - 1)
    INDArray scoreArr = output.rsubi(labels); // Ytrue - Ypred
    scoreArr = scoreArr.divi(DELTA); // score / d
    scoreArr = scoreArr.muli(scoreArr); // score^2
    scoreArr = scoreArr.addi(1.); // 1 + score
    scoreArr = Transforms.sqrt(scoreArr, false); // sqrt(score)
    scoreArr = scoreArr.subi(1.); // score - 1
    scoreArr = scoreArr.muli(Math.pow(DELTA, 2)); // d^2 * score
  
    //Weighted loss function
    if (weights != null) {
      if (weights.length() != output.size(1)) {
        throw new IllegalStateException("Weights vector (length " + weights.length()
          + ") does not match output.size(1)=" + output.size(1));
      }
      scoreArr.muliRowVector(weights.castTo(scoreArr.dataType()));
    }
    
    //Loss function with masking
    if (mask != null) {
      LossUtil.applyMask(scoreArr, mask);
    }
    return scoreArr;
  }
  
  @Override
  public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                             boolean average) {
    INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
    
    double score = scoreArr.sumNumber().doubleValue();
    
    if (average)
      score /= scoreArr.size(0);
    
    return score;
  }
  
  @Override
  public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
    return scoreArr.sum(true,1);
  }
  
  @Override
  public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    if(!labels.equalShapes(preOutput)){
      Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
    }
    labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
    INDArray output = activationFn.getActivation(preOutput.dup(), true);
    
    //INDArray dLda = output.subi(labels).muli(2); // MSE: 2 * (Ypred - Ytrue)
    
    // derivative huber: (Ypred - Ytrue) / sqrt( ((Ypred - Ytrue)^2 / r^2) +1 )
    INDArray dLda = output.sub(labels); // (Ypred - Ytrue)
    dLda = Transforms.pow(dLda, 2, false); // dl ^ 2
    dLda = dLda.divi(Math.pow(DELTA, 2)); // dl / d^2
    dLda = dLda.addi(1); // dl + 1
    dLda = Transforms.sqrt(dLda, false); // sqrt(dl)
    dLda = dLda.rdivi(output.sub(labels)); // (Ypred - Ytrue) / dl
    
    if (weights != null) {
      dLda.muliRowVector(weights.castTo(dLda.dataType()));
    }
    
    if (mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
      //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
      //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
      //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
      // error prone - but buy us a tiny bit of performance
      LossUtil.applyMask(dLda, mask);
    }
    
    INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst(); //TODO handle activation function parameter gradients
    
    //Loss function with masking
    if (mask != null) {
      LossUtil.applyMask(gradients, mask);
    }
    
    return gradients;
  }
  
  @Override
  public Pair<Double, INDArray> computeGradientAndScore(INDArray labels,
                                                        INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
    //TODO: probably a more efficient way to do this...
    
    return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
      computeGradient(labels, preOutput, activationFn, mask));
  }
  
  
  /**
   * The opName of this function
   *
   * @return
   */
  @Override
  public String name() {
    return toString();
  }
  
  @Override
  public String toString() {
    if (weights == null)
      return "LossHuber()";
    return "LossHuber(weights=" + weights + ")";
  }
}
