package de.datexis.retrieval.encoder;

import de.datexis.annotator.Annotator;
import de.datexis.annotator.AnnotatorComponent;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.IEncoder;
import de.datexis.retrieval.tagger.LSTMSentenceTagger;
import de.datexis.retrieval.tagger.ModelBuilder;
import de.datexis.tagger.Tagger;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This Annotator capsules a Sentence Embedding
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LSTMSentenceAnnotator extends Annotator {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  /** used by XML deserialization */
  public LSTMSentenceAnnotator() {
    super();
  }
  
  public LSTMSentenceAnnotator(Tagger root) {
    super(root);
  }
  
  protected LSTMSentenceAnnotator(AnnotatorComponent comp) {
    super(comp);
  }
  
  @Override
  public LSTMSentenceTagger getTagger() {
    return (LSTMSentenceTagger) super.getTagger();
  }
  
  public void trainModel(Resource trainingSentences) {
    getTagger().trainModel(trainingSentences);
  }
  
  public LSTMSentenceEncoder asEncoder() {
    return new LSTMSentenceEncoder(getTagger());
  }
  
  public static class Builder {
    
    LSTMSentenceAnnotator ann;
    LSTMSentenceTagger tagger;
    
    protected ILossFunction lossFunc = LossFunctions.LossFunction.MCXENT.getILossFunction();
    protected Activation activation = Activation.SOFTMAX;
    IEncoder inputEncoder, targetEncoder;
    
    private int examplesPerEpoch = -1;
    private int maxTimeSeriesLength = -1;
    private int lstmLayerSize = 256;
    private int embeddingLayerSize = 128;
    private double learningRate = 0.01;
    private double dropOut = 0.5;
    private int iterations = 1;
    private int batchSize = 16; // number of Examples until Sample/Test
    private int numEpochs = 1;
    
    private boolean enabletrainingUI = false;
    
    public Builder() {
      tagger = new LSTMSentenceTagger();
      ann = new LSTMSentenceAnnotator(tagger);
    }
    
    public Builder withId(String id) {
      this.tagger.setId(id);
      return this;
    }
  
    public Builder withDataset(String datasetName, WordHelpers.Language lang) {
      ann.getProvenance().setDataset(datasetName);
      ann.getProvenance().setLanguage(lang.toString().toLowerCase());
      return this;
    }
    
    public Builder withLossFunction(LossFunctions.LossFunction lossFunc, Activation activation) {
      this.lossFunc = lossFunc.getILossFunction();
      this.activation = activation;
      return this;
    }
    
    public Builder withLossFunction(ILossFunction lossFunc, Activation activation) {
      this.lossFunc = lossFunc;
      this.activation = activation;
      return this;
    }
    
    public Builder withModelParams(int lstmLayerSize, int embeddingLayerSize) {
      this.lstmLayerSize = lstmLayerSize;
      this.embeddingLayerSize = embeddingLayerSize;
      return this;
    }
    
    public Builder withTrainingParams(double learningRate, double dropOut, int examplesPerEpoch, int batchSize, int numEpochs) {
      this.learningRate = learningRate;
      this.dropOut = dropOut;
      this.examplesPerEpoch = examplesPerEpoch;
      this.batchSize = batchSize;
      this.numEpochs = numEpochs;
      return this;
    }
    
    public Builder withTrainingParams(double learningRate, double dropOut, int examplesPerEpoch, int maxTimeSeriesLength, int batchSize, int numEpochs) {
      this.learningRate = learningRate;
      this.dropOut = dropOut;
      this.examplesPerEpoch = examplesPerEpoch;
      this.batchSize = batchSize;
      this.maxTimeSeriesLength = maxTimeSeriesLength;
      this.numEpochs = numEpochs;
      return this;
    }
    
    public Builder withInputEncoders(String desc, Encoder inputEncoder) {
      this.inputEncoder = inputEncoder;
      tagger.setInputEncoders(inputEncoder);
      ann.getProvenance().setFeatures(desc);
      ann.addComponent(inputEncoder);
      return this;
    }
    
    public Builder withTargetEncoder(Encoder targetEncoder) {
      this.targetEncoder = targetEncoder;
      tagger.setTargetEncoder(targetEncoder);
      ann.addComponent(targetEncoder);
      return this;
    }
    
    public Builder enableTrainingUI(boolean enable) {
      this.enabletrainingUI = enable;
      return this;
    }
    
    public LSTMSentenceAnnotator build() {
      tagger.initializeNetwork(ModelBuilder.buildLSTMSentenceTagger(
        inputEncoder.getEmbeddingVectorSize(),
        lstmLayerSize,
        embeddingLayerSize,
        targetEncoder.getEmbeddingVectorSize(),
        iterations, learningRate, dropOut, lossFunc, activation)
      );
      if(enabletrainingUI) tagger.enableTrainingUI();
      tagger.setEmbeddingLayerSize(embeddingLayerSize);
      tagger.setTrainingParams(examplesPerEpoch, maxTimeSeriesLength, batchSize, numEpochs, true);
      ann.getProvenance().setTask(tagger.getId());
      tagger.setName(ann.getProvenance().toString());
      tagger.appendTrainLog(printParams());
      return ann;
    }
    
    private String printParams() {
      StringBuilder line = new StringBuilder();
      line.append("TRAINING PARAMS: ").append(tagger.getName()).append("\n");
      line.append("\nEncoders:\n");
      for(Encoder e : tagger.getEncoders()) {
        line.append(e.getId()).append("\t").append(e.getClass().getSimpleName()).append("\t").append(e.getEmbeddingVectorSize()).append("\n");
      }
      line.append("\nNetwork Params:\n");
      line.append("BLSTM").append("\t").append(lstmLayerSize).append("\n");
      line.append("EMB").append("\t").append(embeddingLayerSize).append("\n");
      line.append("\nTraining Params:\n");
      line.append("examples per epoch").append("\t").append(examplesPerEpoch).append("\n");
      line.append("max time series length").append("\t").append(maxTimeSeriesLength).append("\n");
      line.append("epochs").append("\t").append(numEpochs).append("\n");
      line.append("iterations").append("\t").append(iterations).append("\n");
      line.append("batch size").append("\t").append(batchSize).append("\n");
      line.append("learning rate").append("\t").append(learningRate).append("\n");
      line.append("dropout").append("\t").append(dropOut).append("\n");
      line.append("loss").append("\t").append(lossFunc.toString()).append("\n");
      line.append("\n");
      return line.toString();
    }
    
  }
  
}
