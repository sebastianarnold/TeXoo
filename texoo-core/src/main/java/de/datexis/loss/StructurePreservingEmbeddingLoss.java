package de.datexis.loss;

import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StructurePreservingEmbeddingLoss implements ILossFunction {

  protected static final Logger log = LoggerFactory.getLogger(StructurePreservingEmbeddingLoss.class);

  public static final int RIDICULOUSLY_LARGE_NUMBER = 9001; // Its over 9000!
  private float margin;
  private float kNegativeExamples;
  private float joinTerm2Weight; // lambda1
  private float structureConstraintXWeight; // lambda2
  private float structureConstraintYWeight; // lambda3

  public StructurePreservingEmbeddingLoss() {
    structureConstraintYWeight = 1;
    structureConstraintXWeight = 1;
    joinTerm2Weight = 1;
    kNegativeExamples = 1;
    margin = 1;
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

    long singleEmbeddingSize = scoreArr.size(1) / 2L;

    // Split vectors
    INDArray x = scoreArr.get(NDArrayIndex.all(), NDArrayIndex.interval(0, singleEmbeddingSize));
    INDArray y = scoreArr.get(NDArrayIndex.all(), NDArrayIndex.interval(singleEmbeddingSize, scoreArr.size(1)));

    INDArray xLabels = labels.get(NDArrayIndex.all(), NDArrayIndex.interval(0, singleEmbeddingSize));
    INDArray yLabels = labels.get(NDArrayIndex.all(), NDArrayIndex.interval(singleEmbeddingSize, labels.size(1)));


    INDArray yContrastive = Nd4j.create(y.shape());
    INDArray xContrastive = Nd4j.create(x.shape());

    long numRows = preOutput.shape()[0];

    // Sample negative examples
    for (long i = 1; i <= numRows * kNegativeExamples; i++) {
      yContrastive.putRow(i - 1, y.getRow(i % numRows));
      xContrastive.putRow(i - 1, x.getRow(i % numRows));
    }


    // Sample Neighbors in original embeddings
    INDArray xNeighbor = sampleNeighborhood(xLabels, xLabels, true);
    INDArray yNeighbor = sampleNeighborhood(yLabels, yLabels, true);
    INDArray xNotANeighbor = sampleNeighborhood(xLabels, xLabels, false);
    INDArray yNotANeighbor = sampleNeighborhood(yLabels, yLabels, false);

    SameDiff graph = buildSameDiffGraph(x, y, yContrastive, xContrastive, xNeighbor, yNeighbor, xNotANeighbor, yNotANeighbor);

    scoreArr = graph.execAndEndResult();

    //multiply with masks, always
    applyMask(mask, scoreArr);

    return scoreArr.broadcast(numRows, 1);
  }

  @NotNull
  private SameDiff buildSameDiffGraph(INDArray xIn, INDArray yIn, INDArray yContrastiveIn, INDArray xContrastiveIn, INDArray xNeighborIn, INDArray yNeighborIn, INDArray xNotANeighborIn, INDArray yNotANeighborIn) {
    SameDiff graph = SameDiff.create();
    SDVariable x = graph.var("x", xIn);
    SDVariable y = graph.var("y", yIn);

    SDVariable yContrastive = graph.var("yContrastive", yContrastiveIn);
    SDVariable xContrastive = graph.var("xContrastive", xContrastiveIn);

    SDVariable xNeighbor = graph.var("xNeighbor", xNeighborIn);
    SDVariable yNeighbor = graph.var("yNeighbor", yNeighborIn);
    SDVariable xNotANeighbor = graph.var("xNotANeighbor", xNotANeighborIn);
    SDVariable yNotANeighbor = graph.var("yNotANeighbor", yNotANeighborIn);

    SDVariable distancesXY = euclideanDistanceByRowDiff(x, y, graph);
    SDVariable distancesXYContrastive = euclideanDistanceByRowDiff(x, yContrastive, graph);
    SDVariable distancesXContrastiveY = euclideanDistanceByRowDiff(xContrastive, y, graph);
    SDVariable distancesXXNotANeighbor = euclideanDistanceByRowDiff(x, xNotANeighbor, graph);
    SDVariable distancesXXANeighbor = euclideanDistanceByRowDiff(x, xNeighbor, graph);
    SDVariable distancesYYNotANeighbor = euclideanDistanceByRowDiff(y, yNotANeighbor, graph);
    SDVariable distancesYYANeighbor = euclideanDistanceByRowDiff(y, yNeighbor, graph);

    SDVariable joinTerm1 = distancesXY.add(margin).sub(distancesXYContrastive);
    SDVariable joinTerm2 = distancesXY.add(margin).sub(distancesXContrastiveY);
    SDVariable structureX = distancesXXANeighbor.add(margin).sub(distancesXXNotANeighbor);
    SDVariable structureY = distancesYYANeighbor.add(margin).sub(distancesYYNotANeighbor);


    SDVariable joinTerm1c = graph.scalarMax(joinTerm1, 0);
    SDVariable joinTerm2c = graph.scalarMax(joinTerm2, 0).mul(joinTerm2Weight);
    SDVariable structureXc = graph.scalarMax(structureX, 0).mul(structureConstraintXWeight);
    SDVariable structureYc = graph.scalarMax(structureY, 0).mul(structureConstraintYWeight);

    SDVariable scoreDiff = joinTerm1c.add(joinTerm2c).add(structureXc).add(structureYc);
    return graph;
  }

  private INDArray sampleNeighborhood(INDArray target, INDArray label, boolean positiveSampling) {
    INDArray allDistancesXX = Transforms.allEuclideanDistances(label, label, 1);

    if (positiveSampling) {
      INDArray eye = Nd4j.eye(allDistancesXX.size(1));
      eye.muli(RIDICULOUSLY_LARGE_NUMBER);
      allDistancesXX.addi(eye);
      allDistancesXX.muli(-1);
    }

    INDArray neighbors = Nd4j.create(target.shape());
    INDArray argMax = allDistancesXX.argMax(0);

    for (long i = 0; i < argMax.length(); i++) {
      long index = argMax.getInt((int) i);
      INDArray scalar = target.getRow(index);
      neighbors.putRow((int) i, scalar);
    }
    return neighbors;
  }

  private SDVariable euclideanDistanceByRowDiff(SDVariable x, SDVariable y, SameDiff graph) {
    SDVariable transposeX = graph.transpose(x);
    SDVariable transposeY = graph.transpose(y);
    return graph.euclideanDistance(transposeX, transposeY, 0);
  }


  @Override
  public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    INDArray output = activationFn.getActivation(preOutput.dup(), true);
    long numRows = preOutput.shape()[0];

    long singleEmbeddingSize = output.size(1) / 2L;


    // Split vectors
    INDArray x = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, singleEmbeddingSize));
    INDArray y = output.get(NDArrayIndex.all(), NDArrayIndex.interval(singleEmbeddingSize, output.size(1)));

    INDArray xLabels = labels.get(NDArrayIndex.all(), NDArrayIndex.interval(0, singleEmbeddingSize));
    INDArray yLabels = labels.get(NDArrayIndex.all(), NDArrayIndex.interval(singleEmbeddingSize, labels.size(1)));

    INDArray yContrastive = Nd4j.create(y.shape());
    INDArray xContrastive = Nd4j.create(x.shape());

    // Sample negative examples
    for (long i = 1; i <= numRows * kNegativeExamples; i++) {
      yContrastive.putRow(i - 1, y.getRow(i % numRows));
      xContrastive.putRow(i - 1, x.getRow(i % numRows));
    }

    // Sample Neighbors in original embeddings
    INDArray xNeighbor = sampleNeighborhood(x, xLabels, true);
    INDArray yNeighbor = sampleNeighborhood(y, yLabels, true);
    INDArray xNotANeighbor = sampleNeighborhood(x, xLabels, false);
    INDArray yNotANeighbor = sampleNeighborhood(y, yLabels, false);

    SameDiff graph = buildSameDiffGraph(x, y, yContrastive, xContrastive, xNeighbor, yNeighbor, xNotANeighbor, yNotANeighbor);

    graph.execBackwardAndEndResult();
    SameDiff gradFn = graph.getFunction("grad");
    INDArray dlDx = gradFn.getArrForVarName("x-grad");
    INDArray dlDy = gradFn.getArrForVarName("y-grad");


    //Everything below remains the same
    output = activationFn.backprop(preOutput.dup(), Nd4j.concat(1, dlDx, dlDy)).getFirst();
    //multiply with masks, always
    applyMask(mask, output);
    return output;
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

  public float getMargin() {
    return margin;
  }

  public StructurePreservingEmbeddingLoss setMargin(int margin) {
    this.margin = margin;
    return this;
  }

  public float getkNegativeExamples() {
    return kNegativeExamples;
  }

  public StructurePreservingEmbeddingLoss setkNegativeExamples(int kNegativeExamples) {
    this.kNegativeExamples = kNegativeExamples;
    return this;
  }

  public float getJoinTerm2Weight() {
    return joinTerm2Weight;
  }

  public StructurePreservingEmbeddingLoss setJoinTerm2Weight(int joinTerm2Weight) {
    this.joinTerm2Weight = joinTerm2Weight;
    return this;
  }

  public float getStructureConstraintXWeight() {
    return structureConstraintXWeight;
  }

  public StructurePreservingEmbeddingLoss setStructureConstraintXWeight(int structureConstraintXWeight) {
    this.structureConstraintXWeight = structureConstraintXWeight;
    return this;
  }

  public float getStructureConstraintYWeight() {
    return structureConstraintYWeight;
  }

  public StructurePreservingEmbeddingLoss setStructureConstraintYWeight(int structureConstraintYWeight) {
    this.structureConstraintYWeight = structureConstraintYWeight;
    return this;
  }
}
