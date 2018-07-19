package de.datexis.models.sector.eval;

import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.tagger.Tagger;
import java.util.Map;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.earlystopping.scorecalc.base.BaseIEvaluationScoreCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Score function for evaluating a MultiLayerNetwork according to an evaluation metric ({@link Evaluation.Metric} such
 * as accuracy, F1 score, etc.
 * Used for both MultiLayerNetwork and ComputationGraph
 *
 * @author Alex Black
 */
public class ClassificationScoreCalculator extends BaseIEvaluationScoreCalculator<Model, ClassificationEvaluation> {

  protected Tagger tagger;
  protected LookupCacheEncoder encoder;

  public ClassificationScoreCalculator(Tagger tagger, LookupCacheEncoder encoder, DataSetIterator iterator){
    super(iterator);
    this.tagger = tagger;
    this.encoder = encoder;
  }

  public ClassificationScoreCalculator(Tagger tagger, LookupCacheEncoder encoder, MultiDataSetIterator iterator){
    super(iterator);
    this.tagger = tagger;
    this.encoder = encoder;
  }

  @Override
  protected ClassificationEvaluation newEval() {
      return new ClassificationEvaluation("score calculation", encoder);
  }

  @Override
    public double calculateScore(Model network) {
    ClassificationEvaluation eval = newEval();

    if(network instanceof MultiLayerNetwork) {
      DataSetIterator i = (iter != null ? iter : new MultiDataSetWrapperIterator(iterator));
      eval = ((MultiLayerNetwork) network).doEvaluation(i, eval)[0];
    } else if(network instanceof ComputationGraph) {
      MultiDataSetIterator i = (iterator != null ? iterator : new MultiDataSetIteratorAdapter(iter));
      // FIXME: we cannot call this with multiple outputs
      //eval = ((ComputationGraph) network).doEvaluation(i, eval)[0];
      evaluate((ComputationGraph)network, eval, i);
      tagger.appendTrainLog(eval.printClassificationAtKStats());
    } else {
      throw new RuntimeException("Unknown model type: " + network.getClass());
    }
    return finalScore(eval);
  }
  
  /**
   * Override evaluation to use average of forward/backward layers in a single score.
   * @param net
   * @param evaluation
   * @param iterator 
   */
  protected void evaluate(ComputationGraph net, ClassificationEvaluation evaluation, MultiDataSetIterator iterator) {

    if(iterator.resetSupported() && !iterator.hasNext()) {
      iterator.reset();
    }

    MultiDataSetIterator iter = iterator.asyncSupported() ? new AsyncMultiDataSetIterator(iterator, 2, true) : iterator;

    /*WorkspaceMode cMode = configuration.getTrainingWorkspaceMode();
    configuration.setTrainingWorkspaceMode(configuration.getInferenceWorkspaceMode());*/

    boolean useRnnSegments = (net.getConfiguration().getBackpropType() == BackpropType.TruncatedBPTT);
    if(useRnnSegments) throw new UnsupportedOperationException("Evaluation with Truncated BPTT is not implemented.");

    while(iter.hasNext()) {
      MultiDataSet data = iter.next();

      if(data.getFeatures() == null || data.getLabels() == null) {
        break;
      }
      // TODO: this is similar to SectorTagger.encodeMatrix()
      net.clear();
      net.clearLayerMaskArrays();
      net.setInputs(data.getFeatures());
      net.setLayerMaskArrays(data.getFeaturesMaskArrays(), data.getLabelsMaskArrays());
      net.validateInput();
      Map<String,INDArray> result = net.feedForward(false, false, true);
      
      INDArray labels = data.getLabels()[0];
      INDArray predicted = null;
      INDArray mask = data.getLabelsMaskArrays()[0];
      if(result.containsKey("target")) {
        predicted = result.get("target");
      } else if(result.containsKey("targetFW")) {
        predicted = result.get("targetFW").dup();
        predicted.addi(result.get("targetBW")).divi(2); // FW/BW average
        // TODO: we might add another softmax here?
      }
      
      evaluation.eval(labels, predicted, mask);
      
    }
    
    net.clear();
    net.clearLayerMaskArrays();

    if(iterator.asyncSupported()) {
      ((AsyncMultiDataSetIterator) iter).shutdown();
    }
    
  }
    
  @Override
  protected double finalScore(ClassificationEvaluation e) {
      return 1. - e.getScore();
  }

}