package de.datexis.cdv;

import de.datexis.annotator.Annotator;
import de.datexis.annotator.AnnotatorComponent;
import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.index.EntityIndex;
import de.datexis.cdv.index.QueryIndex;
import de.datexis.cdv.tagger.CDVModelBuilder;
import de.datexis.cdv.tagger.CDVSentenceIterator;
import de.datexis.cdv.tagger.CDVTagger;
import de.datexis.common.AnnotationHelpers;
import de.datexis.common.Timer;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.IEncoder;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.encoder.impl.FastTextEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.retrieval.index.IVocabulary;
import de.datexis.retrieval.index.InMemoryIndex;
import de.datexis.sector.encoder.ClassEncoder;
import de.datexis.sector.encoder.HeadingEncoder;
import de.datexis.sector.model.SectionAnnotation;
import de.datexis.sector.tagger.SectorEncoder;
import de.datexis.tagger.DocumentSentenceIterator;
import de.datexis.tagger.Tagger;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.AbstractMap;
import java.util.Collection;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * An Annotator that detects sections in a Document and assigns labels.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class CDVAnnotator extends Annotator {

  protected final static Logger log = LoggerFactory.getLogger(CDVAnnotator.class);
  
  protected Timer timer = new Timer();
  
  public CDVAnnotator() {
  }
  
  public CDVAnnotator(Tagger root) {
    super(root);
  }
  
  protected CDVAnnotator(AnnotatorComponent comp) {
    super(comp);
  }
  
  @Override
  public CDVTagger getTagger() {
    return (CDVTagger) super.getTagger();
  }
  
  @JsonIgnore
  public IEncoder getEntityEncoder() {
    return getTagger().getEntityEncoder();
  }
  
  @JsonIgnore
  public IEncoder getAspectEncoder() {
    return getTagger().getAspectEncoder();
  }
  
  @Override
  public void annotate(Collection<Document> docs) {
    annotateSentences(docs);
  }
  
  /**
   * use tagger to generate and attach PRED vectors to Sentences
   */
  public void annotateSentences(Collection<Document> docs) {
    log.info("Running CDV neural net encoding...");
    timer.start();
    getTagger().attachCDVSentenceVectors(docs, DocumentSentenceIterator.Stage.ENCODE);
    timer.stop();
    getTagger().appendTestLog("Encoding complete", timer.getLong());
  }
  
  /**
   * use tagger to generate and attach PRED matrices to Documents
   */
  public void annotateDocuments(Collection<Document> docs) {
    log.info("Running CDV neural net encoding...");
    timer.start();
    getTagger().attachCDVDocumentMatrix(docs);
    timer.stop();
    getTagger().appendTestLog("Encoding complete", timer.getLong());
  }
  
  public void annotateDocumentsBaseline(Collection<Document> docs) {
    // use tagger to generate and attach PRED vectors to Sentences
    log.info("Running CDV baseline encoding...");
    timer.start();
    getTagger().attachMatrixBaseline(docs);
    timer.stop();
    getTagger().appendTestLog("Encoding complete", timer.getLong());
  }
  
  @Deprecated
  public void printPredictions(Dataset test, AspectIndex index, SectionAnnotation.Field field) {
    if(getEntityEncoder().getClass() == FastTextEncoder.class) {
      // development code: Test with plain FastText as index
      for(Document doc : test.getDocuments()) {
        for(Map.Entry<Sentence, SectionAnnotation> ann : AnnotationHelpers.getSpanAnnotationsMap(doc, Sentence.class, SectionAnnotation.class)) {
          System.out.println(ann.getValue().getSectionHeading());
          INDArray exp = encodeAnnotation(getEntityEncoder(), ann.getValue());
          INDArray pred = ann.getKey().getVector(CDVTagger.class);
          System.out.println(((FastTextEncoder) getEntityEncoder()).getNearestNeighbours(exp, 3).toString() + " -> " +
                  ((FastTextEncoder) getEntityEncoder()).getNearestNeighbours(pred, 3).toString());
          //System.out.println(exp.transpose());
          //System.out.println(pred.transpose());
        }
      }
    } else if(index.getClass() == AspectIndex.class) {
      // development code: Test with KNN index over FastText
      for(Document doc : test.getDocuments()) {
        System.out.println();
        System.out.println(doc.getId());
        for(Map.Entry<Sentence, SectionAnnotation> ann : AnnotationHelpers.getSpanAnnotationsMap(doc, Sentence.class, SectionAnnotation.class)) {
          INDArray pred = ann.getKey().getVector(getAspectEncoder().getClass());
          String label = ann.getValue().getAnnotation(field);
          System.out.println(index.getKeyPreprocessor().preProcess(label) + "\t -> " +
                  index.find(pred, 3).toString());
        }
      }
    } else {
      throw new IllegalArgumentException("Target encoder has no evaluation: " + getEntityEncoder().getClass().toString());
    }
  }
  
   /**
   * Add vectors to GOLD and PRED annotations (required for evaluation)
   */
 @Deprecated
  protected static void attachVectorsToAnnotations(Document doc, LookupCacheEncoder targetEncoder) {
    // attach GOLD vectors
    for(SectionAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, SectionAnnotation.class)) {
      if(targetEncoder.getClass() == ClassEncoder.class) {
        INDArray exp = targetEncoder.encode(ann.getSectionLabel());
        ann.putVector(ClassEncoder.class, exp);
      } else if(targetEncoder.getClass() == HeadingEncoder.class) {
        INDArray exp = targetEncoder.encode(ann.getSectionHeading());
        ann.putVector(HeadingEncoder.class, exp);
      }
    }
    // attach PRED vectors and labels from empty Annotations
    for(SectionAnnotation ann : doc.getAnnotations(Annotation.Source.PRED, SectionAnnotation.class)) {
      int count = 0;
      INDArray pred = Nd4j.zeros(targetEncoder.getEmbeddingVectorSize(), 1);
      for(Sentence s : doc.streamSentencesInRange(ann.getBegin(), ann.getEnd(), false).collect(Collectors.toList())) {
        pred.addi(s.getVector(targetEncoder.getClass()));
        count++;
      }
      if(count > 1) pred.divi(count);
      if(targetEncoder.getClass() == ClassEncoder.class) {
        ann.putVector(ClassEncoder.class, pred);
        ann.setSectionLabel(targetEncoder.getNearestNeighbour(pred));
        ann.setConfidence(targetEncoder.getMaxConfidence(pred));
      } else if(targetEncoder.getClass() == HeadingEncoder.class) {
        ann.putVector(HeadingEncoder.class, pred);
        Collection<String> preds = targetEncoder.getNearestNeighbours(pred, 2);
        ann.setSectionHeading(StringUtils.join(preds, "/"));
        ann.setConfidence(targetEncoder.getMaxConfidence(pred));
      }
    }
  }

  public static INDArray encodeAnnotation(IEncoder enc, PassageAnnotation ann) {
    if(enc instanceof FastTextEncoder) {
      return enc.encode(ann.getLabel().replaceAll(" | ", " "));
    } else {
      return enc.encode(ann.getLabel());
    }
  }
  
  /**
   * Encode an average of all given Annotations.
   * TODO: try max instead?
   */
  @Deprecated
  public static INDArray encodeAnnotations(IEncoder enc, Collection<PassageAnnotation> anns) {
    INDArray vec, sum = Nd4j.zeros(DataType.FLOAT, enc.getEmbeddingVectorSize(), 1);
    int count = 0;
    for(PassageAnnotation ann : anns) {
      if(enc instanceof IVocabulary) vec = ((IVocabulary) enc).lookup(ann.getLabel());
      else vec = enc.encode(ann.getLabel());
      if(vec != null && vec.maxNumber().doubleValue() != 0) {
        sum.addi(vec);
        count++;
      } else {
        log.warn("could not encode/lookup '{}'", ann.getLabel());
      }
    }
    return count > 1 ? sum.divi(count) : sum;
  }
  
  /**
   * @return a pair containing the balanced vector from all given Annoations and the weight factor for its label
   */
  public static Map.Entry<INDArray, Double> lookupAnnotations(QueryIndex index, Collection<? extends PassageAnnotation> anns, boolean balancing) {
    INDArray vec, sum = Nd4j.zeros(DataType.FLOAT, index.getEmbeddingVectorSize(), 1);
    int count = 0;
    // put weight on rares examples
    double exampleWeight = balancing ? 0. : 1.;
    for(PassageAnnotation ann : anns) {
      double weight = 1.;
      String label = ann.getLabel();
      if(label.equals("Abstract")) label ="description"; // rewrite this label using a better word
      vec = index.lookup(label);
      if(index instanceof EntityIndex) {
        if(vec == null) {
          //String mention = ann.getDocumentRef().getText(ann);
          //log.debug("fallback encoding entity '{}' ({})", mention, label); // this should not for training
          //vec = index.encode(mention); // fallback encoding
          log.trace("missing encoding for entity '{}'", label); // this should not for training
        }
        if(balancing) {
          weight = (ann.getBegin() == 0 ? 1. : 0.1); // put weight on salient entity
          exampleWeight = Math.max(exampleWeight, weight); // use the maximum weight for the whole example
        }
      } else if(index instanceof AspectIndex) {
        if(vec == null) {
          // try to generate a label for long-tail aspects which are not contained in index
          // log.debug("fallback encoding aspect '{}'", ann.getLabel()); // there will be a lot of them
          vec = index.encode(label); // fallback encoding
        }
        // put weight on rare aspects
        if(balancing) {
          double factor = index.weightFactor(label); // weight balancing for rare labels
          exampleWeight = Math.max(exampleWeight, factor); // frequent examples get downweighted
          weight = 1. + factor; // rare labels get boosted
        }
      }
      // sum all Annotations on this example
      if(vec != null && vec.maxNumber().doubleValue() != 0) {
        sum.addi(vec.muli(weight));
        count++;
      }
    }
    if(count == 0) return new AbstractMap.SimpleEntry<>(null, 0.);
    else return new AbstractMap.SimpleEntry<>(Transforms.unitVec(count > 1 ? sum.divi(count) : sum), exampleWeight);
  }

  /**
   * @return Matrix sentences x layersize that contains target predictions for a Document
   */
  protected static INDArray getLayerMatrix(Document doc, String layerClass) {
    
    Sentence sent = doc.getSentence(0);
    
    // initialize embedding matrix
    long layerSize = sent.getVector(layerClass).length();
    INDArray docWeights = Nd4j.zeros(doc.countSentences(), layerSize);
    
    // fill embedding matrix
    int t = 0;
    for(Sentence s : doc.getSentences()) {
      docWeights.getRow(t++).assign(s.getVector(layerClass));
    }

    return docWeights;
    
  }
  
  protected static INDArray getLayerMatrix(Document doc, Class layerClass) {
    return getLayerMatrix(doc, layerClass.getCanonicalName());
  }
  
  /**
   * @return Matrix sentences x layersize that contains Sector embeddings for a Document
   */
  protected static INDArray getEmbeddingMatrix(Document doc) {
    return getLayerMatrix(doc, SectorEncoder.class);
  }

  public static class Builder {
    
    CDVAnnotator ann;
    CDVTagger tagger;
    
    protected ILossFunction lossFunc = LossFunctions.LossFunction.MCXENT.getILossFunction();
    protected Activation activation = Activation.SOFTMAX;
    IEncoder inputEncoder, flagEncoder, entityEncoder, aspectEncoder;
    
    private int examplesPerEpoch = -1;
    private int maxSentencesPerDoc = -1;
    private int maxWordsPerSentence = -1;
    private int lstmWordLayerSize = 128;
    private int lstmSentenceLayerSize = 5121;
    private int embeddingLayerSize = 128;
    private double learningRate = 0.01;
    private double weightDecay = 0.0001;
    private double dropOut = 0.95;
    private int batchSize = 16; // number of Examples until Sample/Test
    private int numEpochs = 1;
    
    private boolean classBalancing = false;
    private boolean enabletrainingUI = false;
    
    public Builder() {
      tagger = new CDVTagger();
      ann = new CDVAnnotator(tagger);
    }
    
    /**
     * This option is unused in the final version.
     */
    @Deprecated
    public Builder withHierarchicalModel(boolean buildHierarchicalModel) {
      //this.hierarchicalModel = buildHierarchicalModel;
      return this;
    }
  
    public Builder withClassBalancing(boolean classBalancing) {
      this.classBalancing = classBalancing;
      return this;
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
    
    public Builder withModelParams(int lstmWordLayerSize, int lstmSentenceLayerSize, int embeddingLayerSize) {
      this.lstmWordLayerSize = lstmWordLayerSize;
      this.lstmSentenceLayerSize = lstmSentenceLayerSize;
      this.embeddingLayerSize = embeddingLayerSize;
      return this;
    }
        
    public Builder withTrainingParams(double learningRate, double dropOut, double weightDecay, int examplesPerEpoch, int batchSize, int numEpochs) {
      this.learningRate = learningRate;
      this.dropOut = dropOut;
      this.weightDecay = weightDecay;
      this.examplesPerEpoch = examplesPerEpoch;
      this.batchSize = batchSize;
      this.numEpochs = numEpochs;
      return this;
    }
    
    public Builder withTrainingParams(double learningRate, double dropOut, double weightDecay, int batchSize, int numEpochs) {
      this.learningRate = learningRate;
      this.dropOut = dropOut;
      this.weightDecay = weightDecay;
      this.batchSize = batchSize;
      this.numEpochs = numEpochs;
      return this;
    }
    
    public Builder withDatasetLimit(int examplesPerEpoch, int maxSentencesPerDoc, int maxWordsPerSentence) {
      this.examplesPerEpoch = examplesPerEpoch;
      this.maxSentencesPerDoc = maxSentencesPerDoc;
      this.maxWordsPerSentence = maxWordsPerSentence;
      return this;
    }
        
    public Builder withInputEncoders(String desc, Encoder inputEncoder, Encoder flagEncoder) {
      this.inputEncoder = inputEncoder;
      this.flagEncoder = flagEncoder;
      tagger.setInputEncoders(inputEncoder, flagEncoder);
      ann.getProvenance().setFeatures(desc);
      ann.addComponent(inputEncoder);
      ann.addComponent(flagEncoder);
      return this;
    }
    
    public Builder withEntityEncoder(QueryIndex entityEncoder) {
      this.entityEncoder = entityEncoder;
      tagger.setEntityEncoder(entityEncoder);
      ann.addComponent(entityEncoder);
      ann.getProvenance().setName("CDV-E");
      return this;
    }
  
    public Builder withAspectEncoder(QueryIndex aspectEncoder) {
      this.aspectEncoder = aspectEncoder;
      tagger.setAspectEncoder(aspectEncoder);
      ann.addComponent(aspectEncoder);
      ann.getProvenance().setName("CDV-A");
      return this;
    }
  
    public Builder withEntityAspectEncoders(QueryIndex entityEncoder, QueryIndex aspectEncoder) {
      this.entityEncoder = entityEncoder;
      this.aspectEncoder = aspectEncoder;
      tagger.setEntityEncoder(entityEncoder);
      tagger.setAspectEncoder(aspectEncoder);
      ann.addComponent(entityEncoder);
      ann.addComponent(aspectEncoder);
      ann.getProvenance().setName("CDV-EA");
      return this;
    }
    
    public Builder enableTrainingUI(boolean enable) {
      this.enabletrainingUI = enable;
      return this;
    }
    
    public CDVAnnotator build() {
      if(entityEncoder != null && aspectEncoder != null) {
        tagger.setIteratorClass(CDVSentenceIterator.class);
        tagger.initializeNetwork(CDVModelBuilder.buildMultiTaskCDV(
          inputEncoder.getEmbeddingVectorSize(),
          flagEncoder.getEmbeddingVectorSize(),
          lstmSentenceLayerSize,
          embeddingLayerSize,
          entityEncoder.getEmbeddingVectorSize(),
          aspectEncoder.getEmbeddingVectorSize(),
          learningRate, dropOut, weightDecay, lossFunc, activation));
      } else {
        tagger.setIteratorClass(CDVSentenceIterator.class);
        tagger.initializeNetwork(CDVModelBuilder.buildSingleTaskCDV(
          inputEncoder.getEmbeddingVectorSize(),
          flagEncoder.getEmbeddingVectorSize(),
          lstmSentenceLayerSize,
          embeddingLayerSize,
          entityEncoder != null ? entityEncoder.getEmbeddingVectorSize() : aspectEncoder.getEmbeddingVectorSize(),
          learningRate, dropOut, weightDecay, lossFunc, activation));
      }
      if(enabletrainingUI) tagger.enableTrainingUI();
      tagger.setTrainingParams(examplesPerEpoch, maxSentencesPerDoc, batchSize, numEpochs, true, classBalancing);
      tagger.setTrainingLimits(examplesPerEpoch, maxSentencesPerDoc, maxWordsPerSentence);
      tagger.setEmbeddingLayerSize(this.embeddingLayerSize);
      //ann.getProvenance().setTask(tagger.getId());
      tagger.setName(ann.getProvenance().toString());
      tagger.appendTrainLog(printParams());
      if(entityEncoder != null && entityEncoder instanceof InMemoryIndex && ((InMemoryIndex) entityEncoder).getEncoder() instanceof Encoder) {
        ann.addComponent((Encoder)((InMemoryIndex) entityEncoder).getEncoder());
        for(Encoder e : ((Encoder)((InMemoryIndex) entityEncoder).getEncoder()).getEncoders()) {
          ann.addComponent(e);
        }
      }
      if(aspectEncoder != null && aspectEncoder instanceof InMemoryIndex && ((InMemoryIndex) aspectEncoder).getEncoder() instanceof Encoder) {
        ann.addComponent((Encoder)((InMemoryIndex) aspectEncoder).getEncoder());
        for(Encoder e : ((Encoder)((InMemoryIndex) aspectEncoder).getEncoder()).getEncoders()) {
          ann.addComponent(e);
        }
      }
      return ann;
    }
    
    private String printParams() {
      StringBuilder line = new StringBuilder();
      line.append("TRAINING PARAMS: ").append(tagger.getName()).append("\n");
      line.append("\nDataset:\n");
      line.append("File").append("\t").append(ann.getProvenance().getDataset()).append("\n");
      line.append("Language").append("\t").append(ann.getProvenance().getLanguage()).append("\n");
      line.append("\nInput Encoders:\n");
      /*for(Encoder e : tagger.getEncoders()) {
        line.append(e.getId()).append("\t").append(e.getClass().getSimpleName()).append("\t").append(e.getEmbeddingVectorSize()).append("\n");
      }
      line.append("\nTarget Encoders:\n");
      for(Encoder e : tagger.getTargetEncoders()) {
        line.append(e.getId()).append("\t").append(e.getClass().getSimpleName()).append("\t").append(e.getEmbeddingVectorSize()).append("\n");
      }*/
      line.append("\nNetwork Params:\n");
      line.append("LSTM").append("\t").append(lstmWordLayerSize).append("\n");
      line.append("BLSTM").append("\t").append(lstmSentenceLayerSize).append("\n");
      line.append("EMB").append("\t").append(embeddingLayerSize).append("\n");
      line.append("\nTraining Params:\n");
      line.append("examples per epoch").append("\t").append(examplesPerEpoch).append("\n");
      line.append("max time series length").append("\t").append(maxSentencesPerDoc).append("\n");
      line.append("epochs").append("\t").append(numEpochs).append("\n");
      line.append("batch size").append("\t").append(batchSize).append("\n");
      line.append("learning rate").append("\t").append(learningRate).append("\n");
      line.append("dropout").append("\t").append(dropOut).append("\n");
      line.append("weight decay").append("\t").append(weightDecay).append("\n");
      line.append("loss").append("\t").append(lossFunc.toString()).append("\n");
      line.append("\n");
      //System.out.println(line.toString());
      return line.toString();
    }

  }
  
}
