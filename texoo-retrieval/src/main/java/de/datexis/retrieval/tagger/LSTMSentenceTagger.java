package de.datexis.retrieval.tagger;

import com.google.common.collect.Lists;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncodingHelpers;
import de.datexis.encoder.IEncoder;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.tagger.AbstractMultiDataSetIterator;
import de.datexis.tagger.DocumentSentenceIterator;
import de.datexis.tagger.Tagger;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Entity/Aspect embedding based on BLSTM / BLOOM training.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LSTMSentenceTagger extends Tagger {

  protected static final Logger log = LoggerFactory.getLogger(LSTMSentenceTagger.class);
  
  // --- encoders --------------------------------------------------------------
  // encoder for word embedding
  protected IEncoder inputEncoder;
  // target representation encoder
  protected IEncoder targetEncoder;
  
  protected Collection<String> stopWords = Collections.emptySet();
  
  protected final FeedForwardToRnnPreProcessor ff2rnn = new FeedForwardToRnnPreProcessor();

  /** used by XML deserializer */
  public LSTMSentenceTagger() {
    super("EMB");
  };
  
  public LSTMSentenceTagger(String id) {
    super(id);
  }
  
  public LSTMSentenceTagger(Resource modelPath) {
    super(modelPath);
    setId("EMB");
  }

  // --- getters / setters -----------------------------------------------------
  
  @JsonIgnore
  @Override
  public ComputationGraph getNN() {
    return (ComputationGraph) net;
  }

  public void initializeNetwork(ComputationGraph net) {
    this.net = net;
  }

  public void setInputEncoders(IEncoder inputEncoder) {
    this.inputEncoder = inputEncoder;
  }
  
  public void setTargetEncoder(IEncoder targetEncoder) {
    this.targetEncoder = targetEncoder;
  }
  
  public Collection<String> getStopWords() {
    return stopWords;
  }
  
  public void setStopWords(Collection<String> stopWords) {
    this.stopWords = stopWords;
  }
  
  @JsonIgnore
  public IEncoder getInputEncoder() {
    return inputEncoder;
  }
  
  @JsonIgnore
  public IEncoder getTargetEncoder() {
    return targetEncoder;
  }

  @Override
  public List<Encoder> getEncoders() {
    return Lists.newArrayList((Encoder)inputEncoder, (Encoder)targetEncoder);
  }
  
  @Override
  public void setEncoders(List<Encoder> encoders) {
    if(encoders.size() != 2)
      throw new IllegalArgumentException("wrong number of encoders given (expected=3, actual=" + encoders.size() + ")");
    inputEncoder = encoders.get(0);
    targetEncoder = encoders.get(1);
  }
  
  public void trainModel(Resource trainingSentences) {
    trainModel(trainingSentences, numEpochs);
  }
  
  @Override
  public void trainModel(Dataset dataset) {
    throw new UnsupportedOperationException("training from Dataset not implemented, please use a TSV file");
  }

  @Override
  public void tag(Collection<Document> docs) {
    throw new UnsupportedOperationException("not implemented yet");
  }

  protected void trainModel(Resource trainingSentences, int numEpochs) {
    LSTMSentenceTaggerIterator it = new LSTMSentenceTaggerIterator(AbstractMultiDataSetIterator.Stage.TRAIN, inputEncoder, targetEncoder, trainingSentences, "utf-8", WordHelpers.Language.EN, stopWords, true, batchSize);
    timer.start();
    appendTrainLog("Training " + getName() + " for " + numEpochs + " epochs.");
    Nd4j.getMemoryManager().togglePeriodicGc(false);
    for(int i = 1; i <= numEpochs; i++) {
      appendTrainLog("Starting epoch " + i + " of " + numEpochs + "\t" );
      triggerEpochListeners(true, i - 1);
      getNN().fit(it);
      //wrapper.fit(it);
      timer.setSplit("epoch");
      appendTrainLog("Completed epoch " + i + " of " + numEpochs + "\t", timer.getLong("epoch"));
      triggerEpochListeners(false, i - 1);
      if(i < numEpochs) it.reset(); // shuffling may take some time
      Nd4j.getMemoryManager().invokeGc();
    }
    timer.stop();
    appendTrainLog("Training complete", timer.getLong());
    //Nd4j.getMemoryManager().togglePeriodicGc(true);
    setModelAvailable(true);
  }
  @Override
  public void testModel(Dataset dataset) {
    //appendTestLog("Testing " + getName() + " with " + n + " examples in " + batches + " batches.");
    timer.start();
    //attachVectors(dataset.getDocuments(), Stage.TEST, targetEncoder.getClass());
    timer.stop();
    appendTestLog("Testing complete", timer.getLong());
  }
  
  public INDArray encodeSentence(Sentence s) {
    INDArray inputMask = LSTMSentenceTaggerIterator.createMask(Collections.singletonList(s), s.getLength(), Token.class);
    INDArray labelsMask = Nd4j.ones(DataType.FLOAT, 1, 1);
    INDArray input = EncodingHelpers.encodeTimeStepMatrix(Collections.singletonList(s), inputEncoder, s.getLength(), Token.class);
    getNN().setLayerMaskArrays(new INDArray[]{inputMask}, new INDArray[]{labelsMask});
    Map<String,INDArray> weights = getNN().feedForward(new INDArray[]{input}, false, true);
    if(weights.containsKey("embedding")) {
      return weights.get("embedding").transpose();
    } else {
      throw new IllegalStateException("Embedding does not have an embeddding layer");
    }
  }
  
  public INDArray encodeBatch(LabeledSentenceIterator.LabeledSentenceBatch batch) {
    INDArray inputMask = LSTMSentenceTaggerIterator.createMask(batch.sentences, batch.maxSentenceLength, Token.class);
    INDArray labelsMask = LSTMSentenceTaggerIterator.createLabelsMask(batch.sentences, Token.class);
    INDArray input = EncodingHelpers.encodeTimeStepMatrix(batch.sentences, inputEncoder, batch.maxSentenceLength, Token.class);
    getNN().setLayerMaskArrays(new INDArray[]{inputMask}, new INDArray[]{labelsMask});
    Map<String,INDArray> weights = getNN().feedForward(new INDArray[]{input}, false, true);
    if(weights.containsKey("embedding")) {
      return weights.get("embedding");
    } else {
      throw new IllegalStateException("Embedding does not have an embeddding layer");
    }
  }
  
  public INDArray encodeBatchMatrix(List<Sentence> examples) {
    int maxSentenceLength = 1;
    for(Sentence s : examples) {
      maxSentenceLength = Math.max(maxSentenceLength, s.countTokens());
    }
    INDArray inputMask = LSTMSentenceTaggerIterator.createMask(examples, maxSentenceLength, Token.class);
    INDArray labelsMask = LSTMSentenceTaggerIterator.createLabelsMask(examples, Token.class);
    INDArray input = EncodingHelpers.encodeTimeStepMatrix(examples, inputEncoder, maxSentenceLength, Token.class);
    getNN().setLayerMaskArrays(new INDArray[]{inputMask}, new INDArray[]{labelsMask});
    Map<String,INDArray> weights = getNN().feedForward(new INDArray[]{input}, false, true);
    if(weights.containsKey("embedding")) {
      return weights.get("embedding");
    } else {
      throw new IllegalStateException("Embedding does not have an embeddding layer");
    }
  }
  
  public Map<String,INDArray> encodeMatrix(DocumentSentenceIterator.DocumentBatch batch) {
    
    MultiDataSet next = batch.dataset;

    Map<String,INDArray> weights = feedForward(getNN(), next);
    
    if(weights.containsKey("embedding")) {
      weights.put("embedding", ff2rnn.preProcess(weights.get("embedding"), batch.size, LayerWorkspaceMgr.noWorkspaces()));
    }

    return weights;
    
  }
  
  public static Map<String,INDArray> feedForward(ComputationGraph net, MultiDataSet next) {
      INDArray[] features = next.getFeatures();
      INDArray[] featuresMasks = next.getFeaturesMaskArrays();
      INDArray[] labelMasks = next.getLabelsMaskArrays();
      net.setLayerMaskArrays(featuresMasks, labelMasks);
      Map<String,INDArray> weights = net.feedForward(features, false, true);
      return weights;
  }
  
  protected void triggerEpochListeners(boolean epochStart, int epochNum){
    Collection<TrainingListener> listeners;
    listeners = getNN().getListeners();
    getNN().getConfiguration().setEpochCount(epochNum);
    if(listeners != null && !listeners.isEmpty()) {
      for(TrainingListener l : listeners) {
        if(epochStart) {
          l.onEpochStart(getNN());
        } else {
          l.onEpochEnd(getNN());
        }
      }
    }
  }
  
  public void enableTrainingUI() {
    /*StatsStorage stats = new InMemoryStatsStorage();
    getNN().addListeners(new StatsListener(stats, 1));
    UIServer.getInstance().attach(stats);
    UIServer.getInstance().enableRemoteListener(stats, true);*/
    throw new UnsupportedOperationException("Training UI is not part of texoo-retrieval. Please use deeplearning4j-ui_2.11 in your code for that.");
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
      ModelSerializer.writeModel(net, os, false);
      setModel(modelFile);
    } catch (IOException ex) {
      log.error(ex.toString());
    } 
  }
  
  @Override
  public void loadModel(Resource modelFile) {
    try(InputStream is = modelFile.getInputStream()) {
      net = ModelSerializer.restoreComputationGraph(is, false); // do not load updater to save memory
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
