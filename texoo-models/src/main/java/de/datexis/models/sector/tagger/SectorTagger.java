package de.datexis.models.sector.tagger;

import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderSet;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.evaluation.ModelEvaluation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.models.sector.eval.ClassificationScoreCalculator;
import de.datexis.models.sector.tagger.DocumentSentenceIterator.Stage;
import de.datexis.tagger.AbstractIterator;
import de.datexis.tagger.Tagger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.shade.jackson.annotation.JsonIgnore; // it is import to use the nd4j version in this class!
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Map;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.databind.JsonNode;

/**
 *
 * @author sarnold
 */
public class SectorTagger extends Tagger {

  protected static final Logger log = LoggerFactory.getLogger(SectorTagger.class);
  
  // sector tagger can only be accessed from one thread at a time
  protected final Object lock = new Object();
  
  // n-hot encoder, such as bag-of-words or trigrams
  protected Encoder bagEncoder = null;
  // embedding encoder from lower layers
  protected Encoder embEncoder = null;
  // flag encoder, such as position
  protected Encoder flagEncoder = null;
  // a single target encoder
  protected Encoder targetEncoder = null;
  
  protected int batchSize = 16;
  protected int numExamples = -1;
  protected int numEpochs = 1;
  protected boolean randomize = true;
  protected int workers = 4;
  
  protected boolean requireSubsampling;
  
  protected int embeddingLayerSize;
  
  protected ModelEvaluation eval = new ModelEvaluation("null");
  private final FeedForwardToRnnPreProcessor ff2rnn = new FeedForwardToRnnPreProcessor();

  /** used by XML deserializer */
  public SectorTagger() {
    super("SECTOR");
  };
  
  public SectorTagger(Resource modelPath) {
    super(modelPath);
    setId("SECTOR");
  }

  @JsonIgnore
  public ComputationGraph getNN() {
    return (ComputationGraph) net;
  }

  public boolean isRequireSubsampling() {
    return requireSubsampling;
  }

  public void setRequireSubsampling(boolean requireSubsampling) {
    this.requireSubsampling = requireSubsampling;
  }
  
  public void setInputEncoders(Encoder bagEncoder, Encoder embEncoder, Encoder flagEncoder) {
    this.bagEncoder = bagEncoder;
    this.embEncoder = embEncoder;
    this.flagEncoder = flagEncoder;
  }
  
  public void setTargetEncoder(Encoder targetEncoder) {
    this.targetEncoder = targetEncoder;
  }

  public SectorTagger setTrainingParams(int examplesPerEpoch, int batchSize, int numEpochs, boolean randomize) {
    this.numExamples = examplesPerEpoch;
    this.batchSize = batchSize;
    this.numEpochs = numEpochs;
    this.randomize = randomize;
    return this;
  }
  
  public SectorTagger setWorkspaceParams(int workers) {
    this.workers = workers;
    return this;
  }

  @Override
  @JsonIgnore
  public EncoderSet getEncoders() {
    // FIXME: better return a map <role,encoder>
    return new EncoderSet(bagEncoder, embEncoder, flagEncoder);
  }

  @Override
  public void addInputEncoder(Encoder e) {
    if(bagEncoder == null) bagEncoder = e;
    else if(embEncoder == null) embEncoder = e;
    else if(flagEncoder == null) flagEncoder = e;
    else throw new IllegalArgumentException("all three input encoders are already set");
  }

  @Override
  @JsonIgnore
  public EncoderSet getTargetEncoders() {
    return new EncoderSet(targetEncoder);
  }

  @JsonIgnore
  public Encoder getTargetEncoder() {
    return targetEncoder;
  }
  
  @Override
  public void addTargetEncoder(Encoder e) {
    if(targetEncoder == null) targetEncoder = e;
    else throw new IllegalArgumentException("target encoder is already set");
  }
  
  public void setEval(ModelEvaluation eval) {
    this.eval = eval;
  }
  
  @Override
  public SectorTagger setEncoders(EncoderSet encoders) {
    this.encoders = encoders;
    this.inputVectorSize = encoders.getVectorSize();
    return this;
  }
  
  @Override
  @Deprecated
  public void addEncoder(Encoder e) {
    // FIXME: we should add input and target encodersets to the XML
    throw new UnsupportedOperationException("multi encoders not implemented yet.");
  }

  public int getBatchSize() {
    return batchSize;
  }

  public void setBatchSize(int batchSize) {
    this.batchSize = batchSize;
  }

  public int getNumEpochs() {
    return numEpochs;
  }

  public void setNumEpochs(int numEpochs) {
    this.numEpochs = numEpochs;
  }

  public boolean isRandomize() {
    return randomize;
  }

  public void setRandomize(boolean randomize) {
    this.randomize = randomize;
  }

  public int getEmbeddingLayerSize() {
    return embeddingLayerSize;
  }

  public void setEmbeddingLayerSize(int embeddingLayerSize) {
    this.embeddingLayerSize = embeddingLayerSize;
  }
  
  public SectorTagger buildMultiFwBwSectorNetwork(int ffwLayerSize, int lstmLayerSize, int embeddingLayerSize, int iterations, double learningRate, double dropout, ILossFunction lossFunc, Activation activation) {
    log.info("initializing graph with layer sizes bag={}, lstm={}, emb={} and {} loss", ffwLayerSize, lstmLayerSize, embeddingLayerSize, lossFunc.name());
    
    // size of the concatenated input vector (after FF layers)
    int sentenceVectorSize;
    this.embeddingLayerSize = embeddingLayerSize;
    
    ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam(learningRate))
        .l2(0.0001)
        .trainingWorkspaceMode(WorkspaceMode.SINGLE) // SINGLE faster vs. SEPARATE less memory - use SEPARATE for CUDA...?
        .inferenceWorkspaceMode(WorkspaceMode.SINGLE) // NONE required because of async access of sub-models in SectorTaggerIterator
				.graphBuilder()
    // INPUT LAYERS
        .addInputs("bag")
        .addInputs("emb")
        .addInputs("flag");
    // FF LAYERS
    if(ffwLayerSize > 0) {
      sentenceVectorSize = ffwLayerSize + embEncoder.getVectorSize() + flagEncoder.getVectorSize();
      gb.addLayer("FF1", new DenseLayer.Builder()
            .nIn(bagEncoder.getVectorSize()).nOut(ffwLayerSize)
            .activation(Activation.ELU)
            .weightInit(WeightInit.RELU)
            .build(), "bag")
        .addLayer("FF2", new DenseLayer.Builder()
            .nIn(ffwLayerSize).nOut(ffwLayerSize)
            .activation(Activation.ELU)
            .weightInit(WeightInit.RELU)
            .build(), "FF1")
        .addVertex("surf", new PreprocessorVertex(new FeedForwardToRnnPreProcessor()), "FF2")
        .addVertex("sentence", new MergeVertex(), "surf", "emb", "flag");
    } else {
      sentenceVectorSize = bagEncoder.getVectorSize() + embEncoder.getVectorSize() + flagEncoder.getVectorSize();
      gb.addVertex("sentence", new MergeVertex(), "bag", "emb", "flag");
    }
    // LSTM LAYERS
      gb.addLayer("BLSTM", new Bidirectional(Bidirectional.Mode.CONCAT, new GravesLSTM.Builder()
          .nIn(sentenceVectorSize).nOut(lstmLayerSize)
          .activation(Activation.TANH)
          .gateActivationFunction(Activation.SIGMOID)
          .dropOut(dropout)
          .weightInit(WeightInit.XAVIER)
          .build()), "sentence");
      //gb.addVertex("BLSTMFW", new PreprocessorVertex(new RnnToFeedForwardPreProcessor()), "BLSTM");
      gb.addVertex("FW", new SubsetVertex(0, lstmLayerSize - 1), "BLSTM");
      gb.addVertex("BW", new SubsetVertex(lstmLayerSize, (2 * lstmLayerSize) - 1), "BLSTM");
    // EMBEDDING LAYER
    if(this.embeddingLayerSize > 0) {
      gb.addLayer("embeddingFW", new DenseLayer.Builder()
            .nIn(lstmLayerSize).nOut(embeddingLayerSize)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .build(), "FW")
        //.addVertex("embeddingFW", new L2NormalizeVertex(new int[] {}, 1e-6), "bottleneckFW")
        .addLayer("embeddingBW", new DenseLayer.Builder()
            .nIn(lstmLayerSize).nOut(embeddingLayerSize)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .build(), "BW");
        //.addVertex("embeddingBW", new L2NormalizeVertex(new int[] {}, 1e-6), "bottleneckBW");
        //.addVertex("prev", new LastTimeStepVertex("bag"), "target")
      //gb.addVertex("embedding", new ElementWiseVertex(ElementWiseVertex.Op.Average), "embeddingFW", "embeddingBW").allowDisconnected(true);
      gb.addLayer("targetFW", new RnnOutputLayer.Builder(lossFunc)
            .nIn(embeddingLayerSize).nOut(targetEncoder.getVectorSize())
            .activation(activation)
            .weightInit(WeightInit.XAVIER)
            .build(), "embeddingFW")
        .addLayer("targetBW", new RnnOutputLayer.Builder(lossFunc)
            .nIn(embeddingLayerSize).nOut(targetEncoder.getVectorSize())
            .activation(activation)
            .weightInit(WeightInit.XAVIER)
            .build(), "embeddingBW");
    } else {
      gb.addLayer("targetFW", new RnnOutputLayer.Builder(lossFunc)
            .nIn(lstmLayerSize).nOut(targetEncoder.getVectorSize())
            .activation(activation)
            .weightInit(WeightInit.XAVIER)
            .build(), "FW")
        .addLayer("targetBW", new RnnOutputLayer.Builder(lossFunc)
            .nIn(lstmLayerSize).nOut(targetEncoder.getVectorSize())
            .activation(activation)
            .weightInit(WeightInit.XAVIER)
            .build(), "BW");
      }
      //gb.allowDisconnected(true);
     // gb.addVertex("target", new ElementWiseVertex(ElementWiseVertex.Op.Average), "targetFW", "targetBW");
      // OUTPUT LAYER
      gb.setOutputs("targetFW", "targetBW")
        .setInputTypes(InputType.recurrent(inputVectorSize), InputType.recurrent(inputVectorSize), InputType.recurrent(inputVectorSize))
				.pretrain(false).backprop(true).backpropType(BackpropType.Standard);

    ComputationGraphConfiguration conf = gb.build();
		ComputationGraph lstm = new ComputationGraph(conf);
		lstm.init();
    net = lstm;
		return this;
    
  }
  
  @Override
  public void trainModel(Dataset dataset) {
    SectorTaggerIterator it = new SectorTaggerIterator(Stage.TRAIN, dataset.getDocuments(), this, numExamples, batchSize, true, requireSubsampling);
    trainWithIterator(it, it.numExamples);
  }

  /*public void trainModel(Resource persistedDatasetPath, boolean fetchDataAsync) {
    MultiDataSetFileIterator fileIterator = new MultiDataSetFileIterator(persistedDatasetPath, batchSize);
    MultiDataSetIterator multiDataSetIterator = fetchDataAsync ? new AsyncMultiDataSetIterator(fileIterator) : fileIterator;
    trainWithIterator(multiDataSetIterator, fileIterator.numExamples());
  }*/

  private void trainWithIterator(MultiDataSetIterator it, int numExamples) {
    int batches = numExamples / batchSize;
    timer.start();
    appendTrainLog("Training " + getName() + " with " + numExamples + " examples in " + batches + " batches for " + numEpochs + " epochs.");
    // ParallelWrapper will take care of load balancing between GPUs.
    /*ParallelWrapper wrapper = new ParallelWrapper.Builder(net)
        .prefetchBuffer(24)  // DataSets prefetching options. Set this value with respect to number of actual devices
        .workers(4)          // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
        .averagingFrequency(1) // rare averaging improves performance, but might reduce model accuracy
        .reportScoreAfterAveraging(true) // if set to TRUE, on every averaging model score will be reported
        .build();*/
    trainForEpochs(it, numExamples);
    timer.stop();
    appendTrainLog("Training complete", timer.getLong());
    setModelAvailable(true);
  }

  public EarlyStoppingResult<ComputationGraph> trainModel(Dataset train, Dataset validation, EarlyStoppingConfiguration conf) {
    SectorTaggerIterator trainIt = new SectorTaggerIterator(Stage.TRAIN, train.getDocuments(), this, numExamples, batchSize, true, requireSubsampling);
    SectorTaggerIterator validationIt = new SectorTaggerIterator(Stage.TEST, validation.getDocuments(), this, batchSize, false, requireSubsampling);
    int batches = trainIt.numExamples / batchSize;
    timer.start();
    appendTrainLog("Training " + getName() + " with " + trainIt.numExamples + " examples in " + batches + " batches using early stopping.");
    conf.setScoreCalculator(new ClassificationScoreCalculator(this, (LookupCacheEncoder) targetEncoder, validationIt));
    EarlyStoppingListener<ComputationGraph> listener = new EarlyStoppingListener<ComputationGraph>() {
      @Override
      public void onStart(EarlyStoppingConfiguration<ComputationGraph> conf, ComputationGraph net) {
      }
      @Override
      public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<ComputationGraph> conf, ComputationGraph net) {
        log.info("Finished epoch {} with score {}", epochNum, 1. - score);
      }
      @Override
      public void onCompletion(EarlyStoppingResult<ComputationGraph> result) {
        log.info("Finished training with result {}", result.toString());
      }
    };
    
    //EarlyStoppingParallelTrainer trainer = new EarlyStoppingParallelTrainer(conf, getNN(), null, trainIt, listener, 4, 4, 1, false, false);
    EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(conf, getNN(), trainIt, listener);
    
    EarlyStoppingResult<ComputationGraph> result = trainer.fit();
    timer.stop();
    appendTrainLog("Training complete", timer.getLong());
    net = result.getBestModel();
    setModelAvailable(true);
    return result;
  }
  
  private void trainForEpochs(MultiDataSetIterator it, int numExamples) {
    int n = 0;
    for(int i = 1; i <= numEpochs; i++) {
      getNN().fit(it);
      //wrapper.fit(it);
      n += numExamples;
      timer.setSplit("epoch");
      appendTrainLog("Completed epoch " + i + " of " + numEpochs + "\t" + n, timer.getLong("epoch"));
      it.reset();
    }
  }

  @Override
  public void testModel(Dataset dataset) {
    //appendTestLog("Testing " + getName() + " with " + n + " examples in " + batches + " batches.");
    timer.start();
    attachVectors(dataset.getDocuments(), Stage.TEST, targetEncoder.getClass(), false);
    timer.stop();
    appendTestLog("Testing complete", timer.getLong());
  }
  
  @Override
  public void tag(Collection<Document> docs) {
    throw new UnsupportedOperationException("not implemented");
  }
  
  public Map<String,INDArray> encodeMatrix(DocumentSentenceIterator.DocumentBatch batch) {
    MultiDataSet data = batch.dataset;
    // only one thread can access at a time (use WorkspaceMode.NONE!)
    synchronized(lock) {
      getNN().clear();
      getNN().clearLayerMaskArrays();
      getNN().setInputs(data.getFeatures());
      getNN().setLayerMaskArrays(data.getFeaturesMaskArrays(), data.getLabelsMaskArrays());
      getNN().validateInput();
      Map<String,INDArray> result = getNN().feedForward(false, false, true);
      if(result.containsKey("target")) {
        //predicted = result.get("target");
      } else if(result.containsKey("targetFW")) {
        INDArray fw = result.get("targetFW").dup();
        INDArray bw = result.get("targetBW").dup();
        //result.put("target", Transforms.sqrt(fw.mul(fw).add(bw.mul(bw)).div(2.))); // geometric mean
        result.put("target", fw.add(bw).divi(2)); // average
      }
      if(result.containsKey("embedding")) {
        // old model without FW/BW
        result.put("embedding", ff2rnn.preProcess(result.get("embedding"), batch.size, LayerWorkspaceMgr.noWorkspaces()));
      } else if(result.containsKey("embeddingFW")) {
        // merge FW/BW layers for embedding
        INDArray fw = ff2rnn.preProcess(result.get("embeddingFW"), batch.size, LayerWorkspaceMgr.noWorkspaces());
        INDArray bw = ff2rnn.preProcess(result.get("embeddingBW"), batch.size, LayerWorkspaceMgr.noWorkspaces());
        result.put("embeddingFW", fw);
        result.put("embeddingBW", bw);
        //result.put("embedding", Transforms.sqrt(fw.mul(fw).add(bw.mul(bw)).div(2.))); // geometric mean
        result.put("embedding", fw.add(bw).divi(2)); // average
      }
      return result;
    }
  }
  
  public void attachVectors(Collection<Document> docs, Stage stage, Class<? extends Encoder> targetClass, boolean alignFWBWlayers) {
    
    SectorTaggerIterator it = new SectorTaggerIterator(stage, docs, this, batchSize, false, requireSubsampling);
  
    // label batches of documents
    while(it.hasNext()) {
      attachVectors(it.nextDocumentBatch(), targetClass, alignFWBWlayers);
    }
    
  }
  
  public void attachVectors(DocumentSentenceIterator.DocumentBatch batch, Class<? extends Encoder> targetClass, boolean alignFWBWlayers) {
    Map<String,INDArray> weights = encodeMatrix(batch);
    INDArray target = weights.get("target"); // attach target class vectors
    INDArray targetFW = null, targetBW = null, embeddingFW = null, embeddingBW = null, embedding = null;
    if(weights.containsKey("embedding")) {
      embedding = weights.get("embedding"); // SECTOR embedding [16SxH] -> [16xHxS]
    }
    if(weights.containsKey("embeddingFW")) {
      embeddingFW = weights.get("embeddingFW"); // attach target class vectors
      embeddingBW = weights.get("embeddingBW"); // attach target class vectors
    }
    if(weights.containsKey("targetFW")) {
      targetFW = weights.get("targetFW"); // attach target class vectors
      targetBW = weights.get("targetBW"); // attach target class vectors
    }
    // append vectors to sentences
    int batchNum = 0; for(Document doc : batch.docs) {
      int t = 0; int max = doc.countSentences() - 1;
      for(Sentence s : doc.getSentences()) {
        s.putVector(targetEncoder.getClass(), target.getRow(batchNum).getColumn(t));
        if(alignFWBWlayers == false && embedding != null) {
          s.putVector(SectorEncoder.class, embedding.getRow(batchNum).getColumn(t));
        }
        if(embeddingFW != null) {
          INDArray fw, bw;
          if(alignFWBWlayers) {
            fw = embeddingFW.getRow(batchNum).getColumn(Math.max(0, t-1));
            bw = embeddingBW.getRow(batchNum).getColumn(Math.min(max, t+1));
            // element-wise geometric mean of shifted layers
            s.putVector(SectorEncoder.class, Transforms.sqrt(fw.mul(fw).add(bw.mul(bw)).div(2.)));
          } else {
            fw = embeddingFW.getRow(batchNum).getColumn(t);
            bw = embeddingBW.getRow(batchNum).getColumn(t);
            // keep existing means
          }
          s.putVector("embeddingFW", fw);
          s.putVector("embeddingBW", bw);
        }
        if(targetFW != null) {
          INDArray fw, bw;
          if(alignFWBWlayers) {          
            fw = targetFW.getRow(batchNum).getColumn(Math.max(0, t-1));
            bw = targetBW.getRow(batchNum).getColumn(Math.min(max, t+1));
            // element-wise geometric mean of shifted layers
            s.putVector(targetEncoder.getClass(), Transforms.sqrt(fw.mul(fw).add(bw.mul(bw)).div(2.)));
          } else {
            fw = targetFW.getRow(batchNum).getColumn(t);
            bw = targetBW.getRow(batchNum).getColumn(t);
            // keep existing means
          }
          s.putVector("targetFW", fw);
          s.putVector("targetBW", bw);
        }
        t++;
      }
      batchNum++;
    }
  }
  
  /*public Resource exportPreprocessedDataset(Dataset dataset) {
    SectorTaggerIterator it = new SectorTaggerIterator(Stage.TRAIN, dataset, this, 1, true, requireSubsampling);
    Resource outputPathResource = new SectorTaggerDatasetExporter().exportUsingJavaSerialization(getName(), it);
    log.info("exported data set ({}) to {}", dataset.getName(), outputPathResource.getPath().toString());
    return outputPathResource;
  }*/
  
  public void enableTrainingUI() {
    StatsStorage stats = new InMemoryStatsStorage();
    net.addListeners(new StatsListener(stats, 1));
    UIServer.getInstance().attach(stats);
    UIServer.getInstance().enableRemoteListener(stats, true);
  }
  
  /**
   * Saves the model to <name>.bin.gz
   * @param modelPath
   * @param name 
   */
  @Override
  public void saveModel(Resource modelPath, String name) {
    Resource modelFile = modelPath.resolve(name + ".zip");
    try(OutputStream os = modelFile.getOutputStream()){
      ModelSerializer.writeModel(net, os, true);
      setModel(modelFile);
    } catch (IOException ex) {
      log.error(ex.toString());
    } 
  }
  
  @Override
  public void loadModel(Resource modelFile) {
    try(InputStream is = modelFile.getInputStream()) {
      net = ModelSerializer.restoreComputationGraph(is, true);
    //try(DataInputStream dis = new DataInputStream(modelFile.getInputStream())) {
      // Load parameters from disk:
    //  INDArray newParams = Nd4j.read(dis);
    //  ((MultiLayerNetwork)net).setParameters(newParams);
      setModel(modelFile);
      setModelAvailable(true);
      log.info("loaded Computation Graph from " + modelFile.getFileName());
    } catch(IOException ex) {
      log.error(ex.toString());
    }
  }

  @Override
  public ComputationGraphConfiguration getGraphConfiguration() {
    // overriden, because graph is saved to ZIP
    return null;
  }

  @Override
  public void setGraphConfiguration(JsonNode conf) {
    // overriden, because graph is already loaded from ZIP
  }
  
}
