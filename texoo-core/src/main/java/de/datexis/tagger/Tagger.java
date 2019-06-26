package de.datexis.tagger;

import de.datexis.annotator.AnnotatorComponent;
import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A Deep Neural Network Tagger. This is basically a wrapper for ComputationGraph
 * with references to its Encoders.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class Tagger extends AnnotatorComponent {
  
  protected static final Logger log = LoggerFactory.getLogger(Tagger.class);

  // --- network parameters ----------------------------------------------------

  /** Size of the input vector */
  protected long inputVectorSize;

  /** Size of the embedding vector */
  protected long embeddingVectorSize;

  /** Size of the output vector */
  protected long outputVectorSize;

  private List<Encoder> encoders = new ArrayList<>();

  // --- training parameters ---------------------------------------------------

  protected int batchSize = 16;
  protected int maxTimeSeriesLength = -1;
  protected int numExamples = -1;
  protected int numEpochs = 1;
  protected boolean randomize = true;
  protected int embeddingLayerSize;

  /** The network to train */
  protected Model net;

  // --- constructors ----------------------------------------------------------

  public Tagger(String id) {
    super(false);
    this.id = id;
  }
  
  protected Tagger(long inputVectorSize, long outputVectorSize) {
    super(false);
    this.inputVectorSize = inputVectorSize; // number of inputs (I)
    this.outputVectorSize = outputVectorSize; // number of outputs (K)
  }
  
  protected Tagger(Resource modelFile) {
    super(false);
    loadModel(modelFile);
    //int i = lstm.getnLayers();
    // FIXME: vector sizes!
    //this.inputVectorSize = lstm.getLayerWiseConfigurations().getConf(0).
    //this.outputVectorSize = outputVectorSize; // number of outputs (K)
  }

  // --- property getters / setters --------------------------------------------

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
  
  public int getMaxTimeSeriesLength() {
    return maxTimeSeriesLength;
  }
  
  public void setMaxTimeSeriesLength(int maxTimeSeriesLength) {
    this.maxTimeSeriesLength = maxTimeSeriesLength;
  }
  
  public int getEmbeddingLayerSize() {
    return embeddingLayerSize;
  }

  public void setEmbeddingLayerSize(int embeddingLayerSize) {
    this.embeddingLayerSize = embeddingLayerSize;
  }

  @JsonIgnore
  public Model getNN() {
    return net;
  }

  // --- serialization getters / setters ---------------------------------------

  @Override
  public void setEncoders(List<Encoder> encs) {
    encoders = encs;
    long i = 0;
    for(Encoder e : encs) {
      i += e.getEmbeddingVectorSize();
    }
    inputVectorSize = i;
  }

  /**
   * @return all Encoders (input & output) as Components (not IEncoder)
   */
  @Override
  @JsonIgnore
  public List<Encoder> getEncoders() {
    return encoders;
  }

  public ComputationGraphConfiguration getGraphConfiguration() {
    if(net == null) return null;
    else if(net instanceof ComputationGraph) return ((ComputationGraph) net).getConfiguration();
    else return null;
  }

  public void setGraphConfiguration(JsonNode conf) {
    if(conf != null) {
      String json = conf.toString();
      if(json != null && !json.equals("null")) {
        net = new ComputationGraph(ComputationGraphConfiguration.fromJson(json));
        net.init();
      }
    }
  }

  public MultiLayerConfiguration getLayerConfiguration() {
    if(net == null) return null;
    else if(net instanceof MultiLayerNetwork) return ((MultiLayerNetwork) net).getLayerWiseConfigurations();
    else return null;
  }

  public void setLayerConfiguration(JsonNode conf) {
    if(conf != null) {
      String json = conf.toString();
      if(json != null && !json.equals("null")) {
        net = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json));
        net.init();
      }
    }
  }

  public void setListeners(IterationListener... listeners) {
    net.setListeners(listeners);
  }

  /**
   * @return True, iff all models in all children components are loaded and trained.
   */
  @Override
  @JsonIgnore
  public boolean isModelAvailableInChildren() {
    return encoders.stream().allMatch(child -> child.isModelAvailable());
  }

  // --- builder setters -------------------------------------------------------

  public void setTrainingParams(int examplesPerEpoch, int maxTimeSeriesLength, int batchSize, int numEpochs, boolean randomize) {
    this.numExamples = examplesPerEpoch;
    this.maxTimeSeriesLength = maxTimeSeriesLength;
    this.batchSize = batchSize;
    this.numEpochs = numEpochs;
    this.randomize = randomize;
  }

  /**
   * Saves the model to <name>.bin.gz
   * @param modelPath
   * @param name 
   */
  @Override
  public void saveModel(Resource modelPath, String name) {
    if(net instanceof ComputationGraph) {
      Resource modelFile = modelPath.resolve(name + ".zip");
      try(OutputStream os = modelFile.getOutputStream()){
        ModelSerializer.writeModel(net, os, true);
        setModel(modelFile);
      } catch (IOException ex) {
        log.error(ex.toString());
      } 
    } else if(net instanceof MultiLayerNetwork) {
      Resource modelFile = modelPath.resolve(name + ".bin.gz");
      try(DataOutputStream dos = new DataOutputStream(modelFile.getGZIPOutputStream())){
        // Write the network parameters:
        Nd4j.write(net.params(), dos);
        dos.flush();
        setModel(modelFile);
      } catch (IOException ex) {
        log.error(ex.toString());
      } 
    }
  }
  
  @Override
  public void loadModel(Resource modelFile) {
    if(modelFile.getFileName().endsWith("zip")) {
      try(InputStream is = modelFile.getInputStream()) {
        net = ModelSerializer.restoreComputationGraph(is, true);
        setModel(modelFile);
        setModelAvailable(true);
        log.info("loaded ComputationGraph from " + modelFile.getFileName());
      } catch (IOException ex) {
        log.error(ex.toString());
      }
    } else {
      try(DataInputStream dis = new DataInputStream(modelFile.getInputStream())) {
        INDArray newParams = Nd4j.read(dis);
        ((MultiLayerNetwork)net).setParameters(newParams);
        setModel(modelFile);
        setModelAvailable(true);
        log.info("loaded MultiLayerNetwork from " + modelFile.getFileName());
      } catch (IOException ex) {
        log.error(ex.toString());
      }
    }
  }
  
  @Deprecated
  public void saveUpdater(Resource modelPath, String name) {
    Resource modelFile = modelPath.resolve(name + ".bin.gz");
    INDArray updaterState = null;
    if(net instanceof MultiLayerNetwork) updaterState = ((MultiLayerNetwork) net).getUpdater().getStateViewArray();
    else if(net instanceof ComputationGraph) updaterState = ((ComputationGraph) net).getUpdater().getStateViewArray();
    if(updaterState != null) try(DataOutputStream dos = new DataOutputStream(modelFile.getGZIPOutputStream())){
      Nd4j.write(updaterState, dos);
      dos.flush();
    } catch (IOException ex) {
      log.error(ex.toString());
    } 
  }
  
  public void loadConf(Resource confFile) {
    try {
      // Load network configuration from disk:
      MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(IOUtils.toString(confFile.getInputStream()));
      // Create a MultiLayerNetwork from the saved configuration and parameters
      //confFromJson.setTrainingWorkspaceMode(WorkspaceMode.SINGLE);
      //confFromJson.setInferenceWorkspaceMode(WorkspaceMode.SINGLE);
      net = new MultiLayerNetwork(confFromJson);
      net.init();
    } catch (IOException ex) {
      log.error(ex.toString());
    }
  }

  public void trainModel(Dataset train) {
    throw new UnsupportedOperationException("Training not implemented");
  }

  public void testModel(Dataset dataset) {
    throw new UnsupportedOperationException("Testing not implemented");
  }

  @Deprecated // should be named attachVectors()
  public void tag(Stream<Document> docs) {
    tag(docs.collect(Collectors.toList()));
  }

  @Deprecated // should be named attachVectors()
  public abstract void tag(Collection<Document> docs);

}
