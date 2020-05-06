package de.datexis.cdv.tagger;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
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
 * Builder class that holds network configurations for Heatmap.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class CDVModelBuilder {

  // --- builder functions -----------------------------------------------------
  /**
   * Initialize a new DNN with standard architecture.
   */
  public static ComputationGraph buildSingleTaskCDV(long inputEmbeddingSize, long positionVectorSize, long lstmLayerSize, long embeddingLayerSize, long targetVectorSize, double learningRate, double dropout, double weightDecay, ILossFunction lossFunc, Activation activation) {

    // size of the concatenated input vector (after FF layers)
    long sentenceVectorSize = inputEmbeddingSize + positionVectorSize;

    ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, learningRate, 0.95)))
        .weightInit(WeightInit.XAVIER)
        .weightDecay(weightDecay)
        .dropOut(0)
        .trainingWorkspaceMode(WorkspaceMode.ENABLED)
        .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
        .cacheMode(CacheMode.HOST)
        .graphBuilder()
        
        // INPUT LAYERS - SENTENCE EMBEDDINGS
        .addInputs("input")
        .addInputs("position")
  
        //.addVertex("input_norm", new L2NormalizeVertex(), "input") // input is already normalized
        .addVertex("sentence", new MergeVertex(), "input", "position")
        
        // LSTM LAYERS - SENTENCE LEVEL
        .addLayer("BLSTM", new Bidirectional(Bidirectional.Mode.CONCAT, new LSTM.Builder()
          .nIn(sentenceVectorSize).nOut(lstmLayerSize)
          .activation(Activation.TANH)
          .gateActivationFunction(Activation.SIGMOID)
          .dropOut(dropout)
          .build()), "sentence")
    
        // EMBEDDING LAYER
        .addLayer("bottleneck", new DenseLayer.Builder()
          .nIn(2 * lstmLayerSize).nOut(embeddingLayerSize)
          .activation(Activation.TANH)
          .build(), "BLSTM")
        .addVertex("embedding", new L2NormalizeVertex(new int[]{1}, 1e-8), "bottleneck") // this helps a lot
      
        // OUTPUT LAYER
        .addLayer("target", new RnnOutputLayer.Builder(lossFunc)
          .nIn(embeddingLayerSize).nOut(targetVectorSize)
          .activation(activation)
          .build(), "embedding")
        .setOutputs("target")
        .setInputTypes(InputType.recurrent(inputEmbeddingSize), InputType.recurrent(positionVectorSize))
        .backpropType(BackpropType.Standard);

    ComputationGraphConfiguration conf = gb.build();
    ComputationGraph lstm = new ComputationGraph(conf);
    lstm.init();
    lstm.setListeners(
        new PerformanceListener(16, true)
        //new ScoreIterationListener(16)
    );

    return lstm;

  }
  
  public static ComputationGraph buildMultiTaskCDV(long inputEmbeddingSize, long positionVectorSize, long lstmLayerSize, long embeddingLayerSize, long entityVectorSize, long aspectVectorSize, double learningRate, double dropout, double weightDecay, ILossFunction lossFunc, Activation activation) {
    
    // size of the concatenated input vector (after FF layers)
    long sentenceVectorSize = inputEmbeddingSize + positionVectorSize;
    
    ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, learningRate, 0.975)))
      .weightInit(WeightInit.XAVIER)
      .weightDecay(weightDecay)
      .dropOut(0) // used only in LSTM layers
      .trainingWorkspaceMode(WorkspaceMode.ENABLED)
      .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
      .cacheMode(CacheMode.HOST)
      .graphBuilder()
      
      // INPUT LAYERS - SENTENCE EMBEDDINGS
      .addInputs("input")
      .addInputs("position")
      
      //.addVertex("input_norm", new L2NormalizeVertex(), "input") // input is already normalized
      .addVertex("sentence", new MergeVertex(), "input", "position")
      
      // LSTM LAYERS - SENTENCE LEVEL
      //.addLayer("dropout", new DropoutLayer(dropout), new RnnToFeedForwardPreProcessor(), "sentence")
      .addLayer("BLSTM", new Bidirectional(Bidirectional.Mode.CONCAT, new LSTM.Builder()
        .nIn(sentenceVectorSize).nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .gateActivationFunction(Activation.SIGMOID)
        .dropOut(dropout)
        .build()), "sentence")
      
      // EMBEDDING LAYER
      .addLayer("embedding", new DenseLayer.Builder()
        .nIn(2 * lstmLayerSize).nOut(embeddingLayerSize)
        .activation(Activation.TANH)
        .build(), "BLSTM")
  
      .addLayer("dense_entity", new DenseLayer.Builder()
        .nIn(embeddingLayerSize).nOut(entityVectorSize)
        .activation(Activation.TANH)
        .build(), "embedding")
  
      .addLayer("dense_aspect", new DenseLayer.Builder()
        .nIn(embeddingLayerSize).nOut(aspectVectorSize)
        .activation(Activation.TANH)
        .build(), "embedding")
      
      .addVertex("emb_entity", new L2NormalizeVertex(new int[]{1}, 1e-8), "dense_entity") // this helps a lot
      .addVertex("emb_aspect", new L2NormalizeVertex(new int[]{1}, 1e-8), "dense_aspect") // this helps a lot
      
      // OUTPUT LAYER
      .addLayer("entity", new RnnOutputLayer.Builder(lossFunc)
        .nIn(entityVectorSize).nOut(entityVectorSize)
        .activation(activation)
        .build(), "emb_entity")
  
      .addLayer("aspect", new RnnOutputLayer.Builder(lossFunc)
        .nIn(aspectVectorSize).nOut(aspectVectorSize)
        .activation(activation)
        .build(), "emb_aspect")
      
      .setOutputs("entity", "aspect")
      .setInputTypes(InputType.recurrent(inputEmbeddingSize), InputType.recurrent(positionVectorSize))
      .backpropType(BackpropType.Standard);
    
    ComputationGraphConfiguration conf = gb.build();
    ComputationGraph lstm = new ComputationGraph(conf);
    lstm.init();
    lstm.setListeners(
      new PerformanceListener(32, true),
      new ScoreIterationListener(4)
    );
    
    return lstm;
    
  }
  
}
