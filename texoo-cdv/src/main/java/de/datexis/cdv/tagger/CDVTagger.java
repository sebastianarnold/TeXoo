package de.datexis.cdv.tagger;

import com.google.common.collect.Lists;
import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.index.EntityIndex;
import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncodingHelpers;
import de.datexis.encoder.IEncoder;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.sector.eval.ClassificationScoreCalculator;
import de.datexis.tagger.AbstractMultiDataSetIterator.Stage;
import de.datexis.tagger.DocumentSentenceIterator;
import de.datexis.tagger.Tagger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

/**
 * Heatmap Entity/Aspect encoding (CDV).
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class CDVTagger extends Tagger implements IEncoder {

  protected static final Logger log = LoggerFactory.getLogger(CDVTagger.class);
  
  // --- encoders --------------------------------------------------------------
  // encoder for word or sentence embedding
  protected IEncoder inputEncoder;
  // positional flag encoder
  protected IEncoder flagEncoder;
  // target representation encoder
  protected IEncoder entityEncoder = null;
  // query representation encoder
  protected IEncoder aspectEncoder = null;
  
  protected final FeedForwardToRnnPreProcessor ff2rnn = new FeedForwardToRnnPreProcessor();

  protected int maxWordsPerSentence = -1;
  protected boolean balancing = true;
  
  protected Class<? extends DocumentSentenceIterator> iteratorClass = CDVSentenceIterator.class; // legacy encoder as default
  
  /** used by XML deserializer */
  public CDVTagger() {
    super("HTM");
  };
  
  public CDVTagger(String id) {
    super(id);
  }
  
  public CDVTagger(Resource modelPath) {
    super(modelPath);
    setId("HTM");
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
  
  public void setTrainingParams(int examplesPerEpoch, int maxTimeSeriesLength, int batchSize, int numEpochs, boolean randomize, boolean balancing) {
    super.setTrainingParams(examplesPerEpoch, maxTimeSeriesLength, batchSize, numEpochs, randomize);
    this.balancing = balancing;
  }
  
  public void setTrainingLimits(int examplesPerEpoch, int maxSentencesPerDoc, int maxWordsPerSentence) {
    this.numExamples = examplesPerEpoch;
    this.maxTimeSeriesLength = maxSentencesPerDoc;
    this.maxWordsPerSentence = maxWordsPerSentence;
  }
  
  public void setInputEncoders(IEncoder inputEncoder, IEncoder flagEncoder) {
    this.inputEncoder = inputEncoder;
    this.flagEncoder = flagEncoder;
  }
  
  public void setEntityEncoder(IEncoder entityEncoder) {
    this.entityEncoder = entityEncoder;
  }
  
  public void setAspectEncoder(IEncoder aspectEncoder) {
    this.aspectEncoder = aspectEncoder;
  }
  
  @JsonIgnore
  public IEncoder getEntityEncoder() {
    return entityEncoder;
  }
  
  @JsonIgnore
  public IEncoder getAspectEncoder() {
    return aspectEncoder;
  }
  
  @Override
  public List<Encoder> getEncoders() {
    return Lists.newArrayList((Encoder)inputEncoder, (Encoder)flagEncoder, (Encoder) entityEncoder, (Encoder) aspectEncoder);
  }
  
  @Override
  public void setEncoders(List<Encoder> encoders) {
    if(encoders.size() >= 3) {
      inputEncoder = encoders.get(0);
      flagEncoder = encoders.get(1);
    } else {
      throw new IllegalArgumentException("wrong number of encoders given (expected=3+, actual=" + encoders.size() + ")");
    }
    
    if(encoders.size() == 3) {
      // Single-Task model
      Encoder enc = encoders.get(2);
      if(enc instanceof EntityIndex) {
        entityEncoder = enc;
      } else if(enc instanceof AspectIndex) {
        aspectEncoder = enc;
      } else {
        throw new IllegalArgumentException("got unknown encoder " + enc.getClass().getName());
      }
    } else if(encoders.size() == 4) {
      // Multi-Task model
      entityEncoder = encoders.get(2);
      aspectEncoder = encoders.get(3);
    } else {
      throw new IllegalArgumentException("wrong number of encoders given (expected=3+, actual=" + encoders.size() + ")");
    }
  }
  
  public int getMaxWordsPerSentence() {
    return maxWordsPerSentence;
  }
  
  public void setMaxWordsPerSentence(int maxWordsPerSentence) {
    this.maxWordsPerSentence = maxWordsPerSentence;
  }
  
  public Class<? extends DocumentSentenceIterator> getIteratorClass() {
    return iteratorClass;
  }
  
  public void setIteratorClass(Class<? extends DocumentSentenceIterator> iteratorClass) {
    this.iteratorClass = iteratorClass;
  }
  
  @Override
  public void trainModel(Dataset dataset) {
    trainModel(dataset, numEpochs);
  }

  @Override
  public void tag(Collection<Document> docs) {
    throw new UnsupportedOperationException("not implemented yet");
  }

  protected DocumentSentenceIterator createIterator(Stage stage, Collection<Document> docs) {
    if(iteratorClass.equals(CDVWordIterator.class)) {
      // return Word + Sentence encoder for single task
      if(stage.equals(Stage.TRAIN)) return new CDVWordIterator(stage, docs, this, numExamples, maxTimeSeriesLength, maxWordsPerSentence, batchSize, true, balancing);
      else return new CDVWordIterator(stage, docs, this, -1, -1, maxWordsPerSentence, batchSize, false, balancing);
    } else {
      // return Sentence encoder for single task
      if(stage.equals(Stage.TRAIN)) return new CDVSentenceIterator(stage, docs, this, numExamples, maxTimeSeriesLength, batchSize, true, balancing);
      else return new CDVSentenceIterator(stage, docs, this, -1, -1, batchSize, false, balancing);
    }
  }
  
  protected synchronized void trainModel(Dataset dataset, int numEpochs) {
    DocumentSentenceIterator it = createIterator(Stage.TRAIN, dataset.getDocuments());
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
    int n = 0;
    Nd4j.getMemoryManager().togglePeriodicGc(false);
    for(int i = 1; i <= numEpochs; i++) {
      appendTrainLog("Starting epoch " + i + " of " + numEpochs + "\t" + n);
      triggerEpochListeners(true, i - 1);
      getNN().fit(it);
      //wrapper.fit(it);
      n += numExamples;
      timer.setSplit("epoch");
      appendTrainLog("Completed epoch " + i + " of " + numEpochs + "\t" + n, timer.getLong("epoch"));
      triggerEpochListeners(false, i - 1);
      if(i < numEpochs) it.reset(); // shuffling may take some time
      Nd4j.getMemoryManager().invokeGc();
    }
    timer.stop();
    appendTrainLog("Training complete", timer.getLong());
    //Nd4j.getMemoryManager().togglePeriodicGc(true);
    setModelAvailable(true);
  }
  
  public EarlyStoppingResult<ComputationGraph> trainModel(Dataset train, Dataset validation, EarlyStoppingConfiguration conf) {
    DocumentSentenceIterator trainIt = createIterator(Stage.TRAIN, train.getDocuments());
    DocumentSentenceIterator validationIt = createIterator(Stage.TEST, validation.getDocuments());
    int batches = (int) (trainIt.getNumExamples() / batchSize);
    timer.start();
    appendTrainLog("Training " + getName() + " with " + trainIt.getNumExamples() + " examples in " + batches + " batches using early stopping.");
    conf.setScoreCalculator(new ClassificationScoreCalculator(this, (LookupCacheEncoder) entityEncoder, validationIt));
    EarlyStoppingListener<ComputationGraph> listener = new EarlyStoppingListener<ComputationGraph>() {
      @Override
      public void onStart(EarlyStoppingConfiguration<ComputationGraph> conf, ComputationGraph net) {
        Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();
      }
      @Override
      public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<ComputationGraph> conf, ComputationGraph net) {
        log.info("Finished epoch {} with score {}", epochNum, 1. - score);
        Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();
        Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread().destroyWorkspace();
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
  
  @Override
  public void testModel(Dataset dataset) {
    //appendTestLog("Testing " + getName() + " with " + n + " examples in " + batches + " batches.");
    timer.start();
    attachCDVSentenceVectors(dataset.getDocuments(), Stage.TEST);
    timer.stop();
    appendTestLog("Testing complete", timer.getLong());
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
  
  /**
   * Attach embeddings and output vectors to sentences of each Document.
   */
  public void attachCDVSentenceVectors(Collection<Document> docs, Stage stage) {
    DocumentSentenceIterator it = createIterator(stage, docs);
    while(it.hasNext()) {
      attachCDVSentenceVectors(it.nextDocumentBatch());
    }
  }
  
  protected void attachCDVSentenceVectors(DocumentSentenceIterator.DocumentBatch batch) {
    // predict outputs
    INDArray[] weights;
    synchronized(getNN()) {
      getNN().setLabels(batch.dataset.getLabels()); // setting empty labels because batchsize is calculated from labels array
      weights = getNN().output(false, batch.dataset.getFeatures(), batch.dataset.getFeaturesMaskArrays(), batch.dataset.getLabelsMaskArrays());
    }
      INDArray entityTarget = null, aspectTarget = null, embedding = null;
    if(getEntityEncoder() != null && getAspectEncoder() != null) {
      // Multi-Task model
      entityTarget = weights[0];
      aspectTarget = weights[1];
    } else if(getEntityEncoder() != null) {
      entityTarget = weights[0];
    } else if(getAspectEncoder() != null) {
      aspectTarget = weights[0];
    }
    // Map<String,INDArray> weights = encodeMatrix(batch); // used to access hidden layer as well
    //INDArray target = weights.get("target"); // attach target class vectors
    //INDArray embedding = weights.get("embedding"); // SECTOR embedding [16SxH] -> [16xHxS]
    // append vectors to sentences
    int batchIndex = 0; for(Document doc : batch.docs) {
      int t = 0;
      for(Sentence s : doc.getSentences()) {
        if(entityTarget != null) {
          INDArray sentVec = EncodingHelpers.getTimeStep(entityTarget, batchIndex, t);
          s.putVector(getEntityEncoder().getClass(), sentVec);
        }
        if(aspectTarget != null) {
          INDArray sentVec = EncodingHelpers.getTimeStep(aspectTarget, batchIndex, t);
          s.putVector(getAspectEncoder().getClass(), sentVec);
        }
        if(embedding != null) {
          INDArray embeddingVec = EncodingHelpers.getTimeStep(embedding, batchIndex, t);
          s.putVector(CDVTagger.class, embeddingVec);
        }
        t++;
      }
      batchIndex++;
    }
  }
  
  /**
   * Attach prediction matrix to each Document.
   */
  public void attachCDVDocumentMatrix(Collection<Document> docs) {
    DocumentSentenceIterator it = createIterator(Stage.ENCODE, docs);
    while(it.hasNext()) {
      attachCDVDocumentMatrix(it.nextDocumentBatch());
    }
  }
  
  /**
   * Attach baseline matrix to each Document.
   */
  public void attachMatrixBaseline(Collection<Document> docs) {
    DocumentSentenceIterator it = createIterator(Stage.ENCODE, docs);
    while(it.hasNext()) {
      attachMatrixBaseline(it.nextDocumentBatch());
    }
  }
  
  protected void attachCDVDocumentMatrix(DocumentSentenceIterator.DocumentBatch batch) {
    // predict outputs
    INDArray[] weights;
    synchronized(getNN()) {
      getNN().setLabels(batch.dataset.getLabels()); // setting empty labels because batchsize is calculated from labels array
      weights = getNN().output(false, batch.dataset.getFeatures(), batch.dataset.getFeaturesMaskArrays(), batch.dataset.getLabelsMaskArrays());
    }
    INDArray entityTarget = null, aspectTarget = null, embedding = null;
    if(getEntityEncoder() != null && getAspectEncoder() != null) {
      // Multi-Task model
      entityTarget = weights[0];
      aspectTarget = weights[1];
    } else if(getEntityEncoder() != null) {
      entityTarget = weights[0];
    } else if(getAspectEncoder() != null) {
      aspectTarget = weights[0];
    }
    // append matrices to documents
    int batchIndex = 0; for(Document doc : batch.docs) {
      if(doc.countSentences() > 0) {
        if(entityTarget != null) {
          INDArray docVec = entityTarget.get(point(batchIndex), all(), interval(0, doc.countSentences()));
          // normalize all sentences to Unit length
          for(int i = 0; i < docVec.size(1); i++) {
            INDArray sentVec = docVec.getColumn(i);
            docVec.getColumn(i).assign(Transforms.unitVec(sentVec));
          }
          doc.putVector(getEntityEncoder().getClass(), docVec);
        }
        if(aspectTarget != null) {
          INDArray docVec = aspectTarget.get(point(batchIndex), all(), interval(0, doc.countSentences()));
          // normalize all sentences to Unit length
          for(int i = 0; i < docVec.size(1); i++) {
            INDArray sentVec = docVec.getColumn(i);
            docVec.getColumn(i).assign(Transforms.unitVec(sentVec));
          }
          doc.putVector(getAspectEncoder().getClass(), docVec);
        }
      }
      batchIndex++;
    }
  }
  
  @Deprecated
  protected void attachMatrixBaseline(DocumentSentenceIterator.DocumentBatch batch) {
    // encode outputs (baseline)
    INDArray weights = EncodingHelpers.encodeTimeStepMatrix(batch.docs , entityEncoder, batch.maxDocLength, Sentence.class);
    // append matrices to documents
    int batchIndex = 0; for(Document doc : batch.docs) {
      if(doc.countSentences() > 0) {
        doc.putVector(getEntityEncoder().getClass(), weights.get(point(batchIndex), all(), interval(0, doc.countSentences())));
      }
      batchIndex++;
    }
  }
  
  /**
   * clear layer states to avoid leaks
   */
  @Deprecated
  protected static void clearLayerStates(ComputationGraph net) {
    for(org.deeplearning4j.nn.api.Layer layer : net.getLayers()) {
      layer.clear();
      layer.clearNoiseWeightParams();
    }
    for(org.deeplearning4j.nn.graph.vertex.GraphVertex vertex : net.getVertices()) {
      vertex.clearVertex();
    }
    net.clear();
    net.clearLayerMaskArrays();
  }
  
  public void enableTrainingUI() {
    StatsStorage stats = new InMemoryStatsStorage();
    getNN().addListeners(new StatsListener(stats, 1));
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
  public long getEmbeddingVectorSize() {
    return embeddingLayerSize;
  }

  @Override
  public INDArray encode(String word) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public INDArray encode(Span span) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    throw new UnsupportedOperationException("Not implemented yet.");
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
