package de.datexis.loss;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

public class StructurePreservingEmbeddingLoss implements ILossFunction {


  private int margin;
  private int kNegativeExamples;
  private int joinTerm2Weight; // lambda1
  private int structureConstraintXWeight; // lambda2
  private int structureConstrainYWeight; // lambda3

  public StructurePreservingEmbeddingLoss() {
    structureConstrainYWeight = 1;
    structureConstraintXWeight = 1;
    joinTerm2Weight = 1;
    kNegativeExamples = 1;
  }

  @Override
  public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
    INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

    double score = scoreArr.sumNumber().doubleValue();

    if (average) {
      score /= scoreArr.size(0);
    }

    return score;
  }

  private void applyMask(INDArray mask, INDArray scoreArr) {
    if (mask != null) {
      scoreArr.muliColumnVector(mask);
    }
  }

  @Override
  public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    INDArray scoreArray = scoreArray(labels, preOutput, activationFn, mask);

    return scoreArray.sum(1);
  }

  public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    INDArray scoreArr = activationFn.getActivation(preOutput, true);
    INDArray x = scoreArr.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
    INDArray y = scoreArr.get(NDArrayIndex.all(),NDArrayIndex.interval(2,4));

    INDArray yContrastive = Nd4j.create(y.shape());
    INDArray xContrastive = Nd4j.create(x.shape());

    long numRows = preOutput.shape()[0];
    
    for (long i = 1; i <= numRows * kNegativeExamples; i++) {
      yContrastive.putRow(i - 1, y.getRow(i % numRows));
      xContrastive.putRow(i - 1, x.getRow(i % numRows));
    }


    INDArray xNeighbor = Nd4j.create(x.shape());
    xNeighbor.putRow(0, x.getRow(1));
    xNeighbor.putRow(1, x.getRow(0));
    xNeighbor.putRow(2, x.getRow(3));
    xNeighbor.putRow(3, x.getRow(2));
    
    INDArray yNeighbor = Nd4j.create(y.shape());
    yNeighbor.putRow(0, y.getRow(1));
    yNeighbor.putRow(1, y.getRow(0));
    yNeighbor.putRow(2, y.getRow(3));
    yNeighbor.putRow(3, y.getRow(2));
    
    INDArray xNotANeighbor = Nd4j.create(x.shape());
    xNotANeighbor.putRow(0, x.getRow(3));
    xNotANeighbor.putRow(1, x.getRow(3));
    xNotANeighbor.putRow(2, x.getRow(0));
    xNotANeighbor.putRow(3, x.getRow(0));

    INDArray yNotANeighbor = Nd4j.create(y.shape());
    yNotANeighbor.putRow(0, y.getRow(3));
    yNotANeighbor.putRow(1, y.getRow(3));
    yNotANeighbor.putRow(2, y.getRow(0));
    yNotANeighbor.putRow(3, y.getRow(0));

    INDArray distancesXY = euclideanDistanceByRow(x, y);
    INDArray distancesXYContrastive = euclideanDistanceByRow(x, yContrastive);
    INDArray distancesXContrastiveY = euclideanDistanceByRow(xContrastive, y);
    INDArray distancesXXNotANeighbor = euclideanDistanceByRow(x, xNotANeighbor);
    INDArray distancesXXNeighbor = euclideanDistanceByRow(x, xNeighbor);
    INDArray distancesYYNotANeighbor = euclideanDistanceByRow(y, yNotANeighbor);
    INDArray distancesYYNeighbor = euclideanDistanceByRow(y, yNeighbor);
    margin = 1;
    INDArray add = distancesXY.add(margin);
    INDArray joinTerm1 = add.sub(distancesXYContrastive);
    INDArray joinTerm2 = distancesXY.add(margin).sub(distancesXContrastiveY);
    INDArray structureX = distancesXXNeighbor.add(margin).sub(distancesXXNotANeighbor);
    INDArray structureY = distancesYYNeighbor.add(margin).sub(distancesYYNotANeighbor);
    joinTerm1 = Transforms.max(joinTerm1, 0);
    joinTerm2 = Transforms.max(joinTerm2, 0).mul(joinTerm2Weight);
    structureX = Transforms.max(structureX, 0).mul(structureConstraintXWeight);

    structureY = Transforms.max(structureY, 0).mul(structureConstrainYWeight);
    scoreArr = joinTerm1.add(joinTerm2).add(structureX).add(structureY);


    //multiply with masks, always
    applyMask(mask, scoreArr);
    
    return scoreArr;
  }

  private INDArray euclideanDistanceByRow(INDArray x, INDArray y) {
    return Nd4j.getExecutioner().exec(new EuclideanDistance(x, y, false), 1);
  }


  @Override
  public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    INDArray output = activationFn.getActivation(preOutput.dup(), true);

    INDArray dlDx = computeDlDx(labels, output);

    //Everything below remains the same
    output = activationFn.backprop(preOutput.dup(), dlDx).getFirst();
    //multiply with masks, always
    applyMask(mask, output);
    return null;
  }

  private INDArray computeDlDx(INDArray labels, INDArray output) {
    return null;
  }

  @Override
  public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
    return new Pair<>(
      computeScore(labels, preOutput, activationFn, mask, average),
      computeGradient(labels, preOutput, activationFn, mask));
  }

  @Override
  public String name() {
    return this.getClass().getSimpleName();
  }
}
