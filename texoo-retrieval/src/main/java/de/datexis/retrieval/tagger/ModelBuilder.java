package de.datexis.retrieval.tagger;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

/**
 * Builder class that holds network configurations.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ModelBuilder {

  /**
   * Initialize a new DNN with standard architecture.
   */
  public static ComputationGraph buildLSTMSentenceTagger(long inputEmbeddingSize, long lstmLayerSize, long embeddingLayerSize, long targetVectorSize, int iterations, double learningRate, double dropout, ILossFunction lossFunc, Activation activation) {
    
    // size of the concatenated input vector (after FF layers)
    long sentenceVectorSize = inputEmbeddingSize;
    
    ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, learningRate, 0.95)))
      .weightInit(WeightInit.XAVIER)
      .l2(0.00001)
      //.weightDecay(0.0001)
      .dropOut(0)
      //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .gradientNormalization(GradientNormalization.ClipL2PerLayer)
      .trainingWorkspaceMode(WorkspaceMode.ENABLED)
      .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
      .cacheMode(CacheMode.HOST)
      .graphBuilder()
      
      // INPUT LAYERS - SENTENCE EMBEDDINGS
      .addInputs("input")
      .addVertex("reverse", new ReverseTimeSeriesVertex("input"), "input")
  
      // LSTM LAYERS - SENTENCE LEVEL
      //.addLayer("BLSTM", new Bidirectional(Bidirectional.Mode.CONCAT, new LSTM.Builder()
      .addLayer("LSTM_FW", new LastTimeStep(new LSTM.Builder()
        .nIn(sentenceVectorSize).nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .gateActivationFunction(Activation.SIGMOID)
        .dropOut(dropout)
        .build()), "input")
      
      .addLayer("LSTM_BW", new LastTimeStep(new LSTM.Builder()
        .nIn(sentenceVectorSize).nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .gateActivationFunction(Activation.SIGMOID)
        .dropOut(dropout)
        .build()), "reverse")
      
      .addVertex("merge", new MergeVertex(), "LSTM_FW", "LSTM_BW" )
      
      //.addLayer("maxPooling", new GlobalPoolingLayer(PoolingType.MAX), "FW")
      
      // EMBEDDING LAYER
      .addLayer("embedding", new DenseLayer.Builder()
        .nIn(2 * lstmLayerSize).nOut(embeddingLayerSize)
        .dropOut(0)
        .activation(Activation.TANH)
        .build(), "merge")
  
      //.addVertex("embedding", new L2NormalizeVertex(new int[]{1}, 1e-8), "dense")
  
      // OUTPUT LAYERS
      .addLayer("target", new OutputLayer.Builder(lossFunc)
        .nIn(embeddingLayerSize).nOut(targetVectorSize)
        .activation(activation)
        .dropOut(0)
        .weightInit(WeightInit.SIGMOID_UNIFORM)
        .build(), "embedding")
      .setOutputs("target")
      .setInputTypes(InputType.recurrent(inputEmbeddingSize))
      .backpropType(BackpropType.Standard);
    
    ComputationGraphConfiguration conf = gb.build();
    ComputationGraph lstm = new ComputationGraph(conf);
    lstm.init();
    lstm.setListeners(
      new PerformanceListener(128, true),
      new ScoreIterationListener(16)
    );
    
    return lstm;
    
  }

}
