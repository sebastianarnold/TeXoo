package de.datexis.models.sector;

import de.datexis.models.sector.model.SectionAnnotation;
import de.datexis.annotator.Annotator;
import de.datexis.annotator.AnnotatorComponent;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.models.sector.encoder.ClassEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Annotation.Source;
import de.datexis.tagger.Tagger;
import de.datexis.model.Document;
import de.datexis.model.Dataset;
import de.datexis.model.Sentence;
import de.datexis.models.sector.encoder.ClassTag;
import de.datexis.models.sector.encoder.HeadingEncoder;
import de.datexis.models.sector.encoder.HeadingTag;
import de.datexis.models.sector.eval.SectorEvaluation;
import de.datexis.models.sector.tagger.DocumentSentenceIterator;
import de.datexis.models.sector.tagger.ScoreImprovementMinEpochsTerminationCondition;
import de.datexis.models.sector.tagger.SectorTagger;
import java.util.Collection;
import java.util.Map;
import java.util.stream.Collectors;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Annotator that detects sections in a Document and assigns labels.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class SectorAnnotator extends Annotator {

  protected final static Logger log = LoggerFactory.getLogger(SectorAnnotator.class);
  
  public static enum SegmentationMethod {
    GOLD, // use provided gold standard segmentation (perfect case)
    NEWLINES, // segment at every newline (will produce too many segments)
    TARGET_LABEL, // segment if top-2 labels change
    TARGET_PCA, // segment if top-8 principal components change
    TARGET_MAGNITUDE, // segmentation based on edge detection on target magnitude
    EMBEDDING_MAGNITUDE, // segmentation based on edge detection on embedding magnitude
  };
  
  public SectorAnnotator() {
  }
  
  public SectorAnnotator(Tagger root) {
    super(root);
  }
  
  @Override
  public SectorTagger getTagger() {
    return (SectorTagger) super.getTagger();
  }
  
  public void trainModel(Dataset train) {
    // train tagger
    getTagger().trainModel(train);
  }
  
  /*public void trainModel(Resource pathToPersistedDataSet, WordHelpers.Language lang, String name, boolean useAsyncPrefetch) {
    // configure
    provenance.setDataset(name);
    provenance.setLanguage(lang.toString().toLowerCase());
    getTagger().setName(provenance.toString());
    
    // train tagger
    getTagger().trainModel(pathToPersistedDataSet, useAsyncPrefetch);
  }*/
  
  @Override
  public void annotate(Collection<Document> docs) {
    annotate(docs, SegmentationMethod.NEWLINES);
  }
  
  public void annotate(Collection<Document> docs, SegmentationMethod method) {
    annotate(docs, method, true);
  }
  
  public void annotate(Collection<Document> docs, SegmentationMethod method, boolean alignFwBwEmbeddings) {
    LookupCacheEncoder targetEncoder = (LookupCacheEncoder) getTagger().getTargetEncoder();
    // use tagger to generate and attach PRED vectors to Sentences
    getTagger().attachVectors(docs, DocumentSentenceIterator.Stage.ENCODE, targetEncoder.getClass(), alignFwBwEmbeddings);
    // create Annotations and attach vectors
    switch(method) {
      case GOLD: {
        for(Document doc : docs) detectSectionsFromGold(doc);
      } break;
      case TARGET_LABEL: {
        for(Document doc : docs) detectSectionsFromTargetLabels(doc, targetEncoder, 2);
      } break;
      case TARGET_PCA: {
        for(Document doc : docs) detectSectionsFromTargetPCA(doc, targetEncoder);
      } break;
      case TARGET_MAGNITUDE: {
        for(Document doc : docs) detectSectionsFromTargetMagnitude(doc, targetEncoder);
      } break;
      case EMBEDDING_MAGNITUDE: {
        for(Document doc : docs) detectSectionsFromEmbeddingMagnitude(doc, targetEncoder);
      } break;
      case NEWLINES:
      default: {
        for(Document doc : docs) detectSectionsFromNewlines(doc);
      } break;
    }
    // attach vectors to annotations
    for(Document doc : docs) attachVectorsToAnnotations(doc, targetEncoder);
  }
  
  public double evaluateModel(Dataset test) {
    SectorEvaluation eval;
    if(getTagger().getTargetEncoder().getClass() == HeadingEncoder.class) {
      HeadingEncoder headings = ((HeadingEncoder)getComponent(HeadingEncoder.ID));
      eval = new SectorEvaluation(test.getName(), Annotation.Source.GOLD, Annotation.Source.PRED, headings);
      // we need tags for sentence-level evaluation
      removeTags(test.getDocuments(), Annotation.Source.PRED);
      createHeadingTags(test.getDocuments(), Annotation.Source.GOLD, headings);
      createHeadingTags(test.getDocuments(), Annotation.Source.PRED, headings);
    } else if(getTagger().getTargetEncoder().getClass() == ClassEncoder.class) {
      ClassEncoder classes = ((ClassEncoder)getComponent(ClassEncoder.ID));
      eval = new SectorEvaluation(test.getName(), Annotation.Source.GOLD, Annotation.Source.PRED, classes);
      // we need tags for sentence-level evaluation
      removeTags(test.getDocuments(), Annotation.Source.PRED);
      createClassTags(test.getDocuments(), Annotation.Source.GOLD, classes);
      createClassTags(test.getDocuments(), Annotation.Source.PRED, classes);
    } else {
      throw new IllegalArgumentException("Target encoder has no evaluation: " + getTagger().getTargetEncoder().getClass().toString());
      //conf.setScoreCalculator(new ClassificationScoreMultiCalculator(targetEncoder, Evaluation.Metric.ACCURACY, validationIt));
    }
    // calculate and print scores
    eval.calculateScores(test);
    getTagger().appendTestLog(eval.printDatasetStats(test));
    getTagger().appendTestLog(eval.printEvaluationStats());
    getTagger().appendTestLog(eval.printSingleClassStats());
    return eval.getScore();
  }
  
  public void trainModelEarlyStopping(Dataset train, Dataset validation, int minEpochs, int minEpochsNoImprovement, int maxEpochs) {
    EarlyStoppingConfiguration conf = new EarlyStoppingConfiguration.Builder()
        .evaluateEveryNEpochs(1)
        .epochTerminationConditions(new ScoreImprovementMinEpochsTerminationCondition(minEpochs, minEpochsNoImprovement, maxEpochs))
        .saveLastModel(false)
        .build();
    // train tagger
    EarlyStoppingResult<ComputationGraph> result = getTagger().trainModel(train, validation, conf);
    getTagger().appendTrainLog("Training complete " + result.toString());
  }
    
  protected void createHeadingTags(Iterable<Document> docs, Annotation.Source source, HeadingEncoder headings) {
    HeadingTag.Factory headingTags = new HeadingTag.Factory(headings);
    for(Document doc : docs) {
      if(!doc.isTagAvaliable(source, HeadingTag.class)) {
        if(source.equals(Annotation.Source.GOLD)) headingTags.attachFromSectionAnnotations(doc, source);
        else if(source.equals(Annotation.Source.PRED)) headingTags.attachFromSentenceVectors(doc, HeadingEncoder.class, source);
      }
    }
  }
  
  protected void createClassTags(Iterable<Document> docs, Annotation.Source source, ClassEncoder classes) {
    ClassTag.Factory classTags = new ClassTag.Factory(classes);
    for(Document doc : docs) {
      if(!doc.isTagAvaliable(source, ClassTag.class)) {
        if(source.equals(Annotation.Source.GOLD)) classTags.attachFromSectionAnnotations(doc, source);
        else if(source.equals(Annotation.Source.PRED)) classTags.attachFromSentenceVectors(doc, ClassEncoder.class, source);
      }
    }
  }
  
  protected void removeTags(Iterable<Document> docs, Annotation.Source source) {
    for(Document doc : docs) {
      for(Sentence s : doc.getSentences()) {
        s.clearTags(source);
      }
      doc.setTagAvailable(source, HeadingTag.class, false);
      doc.setTagAvailable(source, ClassTag.class, false);
    }
  }
  
  /**
   * Add vectors to GOLD and PRED annotations (required for evaluation)
   */
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
      INDArray pred = Nd4j.zeros(targetEncoder.getVectorSize(), 1);
      for(Sentence s : doc.streamSentencesInRange(ann.getBegin(), ann.getEnd(), false).collect(Collectors.toList())) {
        pred.addi(s.getVector(ClassEncoder.class));
        count++;
      }
      if(count > 1) pred.divi(count);
      if(targetEncoder.getClass() == ClassEncoder.class) {
        ann.putVector(ClassEncoder.class, pred);
        ann.setSectionLabel(targetEncoder.getNearestNeighbour(pred));
        ann.setConfidence(targetEncoder.getMaxConfidence(pred));
      } else if(targetEncoder.getClass() == HeadingEncoder.class) {
        ann.putVector(HeadingEncoder.class, pred);
        ann.setSectionHeading(targetEncoder.getNearestNeighbours(pred, 3).toString());
        ann.setConfidence(targetEncoder.getMaxConfidence(pred));
      }
    }
  }
  
  /**
   * Add PRED SectionAnnotations from provided gold standard segmentation (perfect case)
   */
  protected static void detectSectionsFromGold(Document doc) {
    SectionAnnotation section = null;
    for(SectionAnnotation ann : doc.getAnnotations(Source.GOLD, SectionAnnotation.class)) {
      section = new SectionAnnotation(Annotation.Source.PRED);
      section.setBegin(ann.getBegin());
      section.setEnd(ann.getEnd());
      doc.addAnnotation(section);
    }
  }
  
  /**
   * Add PRED SectionAnnotation at every newline (will produce too many segments)
   */
  protected static void detectSectionsFromNewlines(Document doc) {
    SectionAnnotation section = null;
    for(Sentence s : doc.getSentences()) {
      boolean endPar = s.streamTokens().anyMatch(t -> t.getText().equals("*NL*") || t.getText().equals("\n"));
      if(section == null) {
        section = new SectionAnnotation(Annotation.Source.PRED);
        section.setBegin(s.getBegin());
      }
      if(endPar) {
        section.setEnd(s.getEnd());
        doc.addAnnotation(section);
        section = null;
      }
    }
    if(section != null) {
      log.warn("found last sentence without newline");
      section.setEnd(doc.getEnd());
      doc.addAnnotation(section);
      section = null;
    }
  }
  
  /**
   * Add PRED Section Annotation based on sentence-wise output predictions.
   * A new segment will start if top label is not contained in previous top-k labels.
   * @param k - the number of labels to check for change (usually 1-3)
   */
  public void detectSectionsFromTargetLabels(Document doc, LookupCacheEncoder targetEncoder, int k) {
    // start first section
    String lastSection = "";
    INDArray sectionPredictions = Nd4j.create(targetEncoder.getVectorSize()).transposei();
    int sectionLength = 0;
    SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
    section.setBegin(doc.getBegin());

    for(Sentence s : doc.getSentences()) {
      INDArray pred = s.getVector(targetEncoder.getClass());
      Collection<String> currentSections = targetEncoder.getNearestNeighbours(pred, k);
      // start new section
      if(!currentSections.contains(lastSection)) {
        if(!lastSection.isEmpty()) doc.addAnnotation(section);
        section = new SectionAnnotation(Annotation.Source.PRED);
        section.setBegin(s.getBegin());
        sectionLength = 0;
        sectionPredictions = Nd4j.create(targetEncoder.getVectorSize()).transposei();
      }
      // update current section
      sectionPredictions.addi(pred);
      sectionLength++;
      String currentSection = targetEncoder.getNearestNeighbour(sectionPredictions.div(sectionLength));
      section.setEnd(s.getEnd());
      lastSection = currentSection;
    }

    // add last section
    if(!lastSection.isEmpty()) doc.addAnnotation(section);
  }
  
  public void detectSectionsFromTargetPCA(Document doc, LookupCacheEncoder targetEncoder) {
    
    int PCA_DIMS = 8;
    
    // preprocess PCA
    INDArray docTargets = Nd4j.zeros(doc.countSentences(), targetEncoder.getVectorSize());
    int t = 0;
    for(Sentence s : doc.getSentences()) {
      docTargets.getRow(t++).assign(s.getVector(targetEncoder.getClass()));
    }
    INDArray docTargetsPCA = docTargets.mmul(PCA.pca_factor(docTargets.dup(), PCA_DIMS, true));
    INDArray docTargetsDelta = deltaMatrix(docTargetsPCA);
    
    // start first section
    int sectionLength = 0;
    SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
    section.setBegin(doc.getBegin());

    t = 0;
    for(Sentence s : doc.getSentences()) {
      // start new section
      if(docTargetsDelta.getDouble(t) >= 1/(double)PCA_DIMS) {
        if(sectionLength > 0) doc.addAnnotation(section);
        section = new SectionAnnotation(Annotation.Source.PRED);
        section.setBegin(s.getBegin());
        sectionLength = 0;
      }
      // update current section
      sectionLength++;
      section.setEnd(s.getEnd());
      t++;
    }

    // add last section
    if(sectionLength > 0) doc.addAnnotation(section);
    
  }
  
  public void detectSectionsFromTargetMagnitude(Document doc, LookupCacheEncoder targetEncoder) {
    
    int PCA_DIMS = 8;
    
    if(doc.countSentences() < 2) return;
      
    // initialize target matrix
    INDArray docTargets = Nd4j.zeros(doc.countSentences(), targetEncoder.getVectorSize());
    int t = 0;
    for(Sentence s : doc.getSentences()) {
      docTargets.getRow(t).assign(s.getVector(targetEncoder.getClass()));
      t++;
    }
    
    INDArray docTargetPCA = gaussianSmooth(docTargets.mmul(PCA.pca_factor(docTargets.dup(), PCA_DIMS, true)));
    INDArray docEdges = detectEdges(docTargetPCA);
    
    // start first section
    int sectionLength = 0;
    SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
    section.setBegin(doc.getBegin());

    t = 0;
    for(Sentence s : doc.getSentences()) {
      // start new section
      if(docEdges.getDouble(t) > 0) {
        if(sectionLength > 0) doc.addAnnotation(section);
        section = new SectionAnnotation(Annotation.Source.PRED);
        section.setBegin(s.getBegin());
        sectionLength = 0;
      }
      // update current section
      sectionLength++;
      section.setEnd(s.getEnd());
      t++;
    }

    // add last section
    if(sectionLength > 0) doc.addAnnotation(section);
    
  }
  
  /**
   * Add PRED SectionAnnotations based on edge detection on embedding magnitude.
   */
  private void detectSectionsFromEmbeddingMagnitude(Document doc, LookupCacheEncoder targetEncoder) {
    
    int PCA_DIMS = 16;
    
    if(doc.countSentences() < 1) return;
    Sentence sent = doc.getSentence(0);
      
    // initialize FW/BW matrices
    int layerSize = sent.getVector("embeddingFW").length();
    INDArray docFW = Nd4j.zeros(doc.countSentences(), layerSize);
    INDArray docBW = Nd4j.zeros(doc.countSentences(), layerSize);
    
    // fill FW/BW matrices
    int t = 0;
    for(Sentence s : doc.getSentences()) {
      docFW.getRow(t).assign(s.getVector("embeddingFW"));
      docBW.getRow(t).assign(s.getVector("embeddingBW"));
      t++;
    }
    
    INDArray docFwPCA = gaussianSmooth(docFW.mmul(PCA.pca_factor(docFW.dup(), PCA_DIMS, true)));
    INDArray docBwPCA = gaussianSmooth(docBW.mmul(PCA.pca_factor(docBW.dup(), PCA_DIMS, true)));
    INDArray docEdges = detectEdges(docFwPCA, docBwPCA);
    
    // start first section
    int sectionLength = 0;
    SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
    section.setBegin(doc.getBegin());

    t = 0;
    for(Sentence s : doc.getSentences()) {
      // start new section
      if(docEdges.getDouble(t) > 0) {
        if(sectionLength > 0) doc.addAnnotation(section);
        section = new SectionAnnotation(Annotation.Source.PRED);
        section.setBegin(s.getBegin());
        sectionLength = 0;
      }
      // update current section
      sectionLength++;
      section.setEnd(s.getEnd());
      t++;
    }

    // add last section
    if(sectionLength > 0) doc.addAnnotation(section);
    
  }
  
  protected static INDArray gaussianSmooth(INDArray target) {
    // gaussian smoothing
    INDArray kernel = Nd4j.zeros(target.rows()).transpose();
    //INDArray docTargetsSmooth = Convolution.convn(docTargetsPCA, kernel, Convolution.Type.FULL);
    INDArray smooth = Nd4j.zerosLike(target);
    // convolution
    for(int t=0; t<kernel.length(); t++) {
      NormalDistribution dist = new NormalDistribution(t, 2.5);
      for(int k=0; k<kernel.length(); k++) {
        kernel.putScalar(k, dist.density(k));
      }
      INDArray conv = target.mulColumnVector(kernel);
      smooth.getRow(t).assign(conv.sum(0));
    }
    return smooth;
  }
  
  /**
   * Returns a matrix [Tx1] that contains cosine distances between forward and backward layer.
   */
  protected static INDArray detectEdges(INDArray fw, INDArray bw) {
    INDArray d1 = Nd4j.zeros(fw.rows(), 1);
    for(int t = 1; t < d1.rows() - 1; t++) {
      double fwd1 = Transforms.cosineDistance(fw.getRow(t), fw.getRow(t+1)); // first derivative - FW is too late
      double bwd1 = Transforms.cosineDistance(bw.getRow(t), bw.getRow(t-1)); // first derivative - BW is too early
      d1.putScalar(t, 0, Math.sqrt(Math.pow(fwd1, 2) + Math.pow(bwd1, 2) / 2.)); // quadratic mean
    }
    INDArray result = Nd4j.zeros(fw.rows(), 1);
    for(int t = 1; t < result.rows() - 1; t++) {
      result.putScalar(t, 0, ((d1.getDouble(t - 1) < d1.getDouble(t)) && (d1.getDouble(t + 1) < d1.getDouble(t))) ? 1 : 0);
    }
    // overwrite first timestep values
    result.putScalar(0, 0, 1);
    return result;
  }
  
  /**
   * Returns a matrix [Tx1] that contains cosine distances between forward and backward layer.
   */
  protected static INDArray detectEdges(INDArray target) {
    INDArray d1 = Nd4j.zeros(target.rows(), 1);
    for(int t = 1; t < d1.rows(); t++) {
      d1.putScalar(t, 0, Transforms.cosineDistance(target.getRow(t), target.getRow(t-1))); // first derivative
    }
    INDArray result = Nd4j.zeros(target.rows(), 1);
    for(int t = 1; t < result.rows() - 1; t++) {
      result.putScalar(t, 0, ((d1.getDouble(t - 1) < d1.getDouble(t)) && (d1.getDouble(t + 1) < d1.getDouble(t))) ? 1 : 0);
    }
    // overwrite first timestep values
    result.putScalar(0, 0, 1);
    return result;
  }
  
  /**
   * Returns a matrix [Tx1] that contains cosine distances between time steps.
   */
  protected static INDArray deltaMatrix(INDArray data) {
    INDArray result = Nd4j.zeros(data.rows(), 1);
    INDArray prev = Nd4j.zeros(data.columns());
    for(int t = 0; t < data.rows(); t++) {
      INDArray vec = data.getRow(t);
      result.putScalar(t, 0, Transforms.cosineDistance(prev, vec));
      prev = vec.dup();
    }
    // overwrite first timestep values with max (might be NaN or too high)
    result.putScalar(0, 0, 1);
    return result;
  }
      
  public static class Builder {
    
    SectorAnnotator ann;
    SectorTagger tagger;
    
    protected Encoder[] encoders = new Encoder[0];
    protected ILossFunction lossFunc = LossFunctions.LossFunction.MCXENT.getILossFunction();
    protected Activation activation = Activation.SOFTMAX;
    protected boolean requireSubsampling = false;
    
    private int examplesPerEpoch = -1;
    private int ffwLayerSize = 0;
    private int lstmLayerSize = 256;
    private int embeddingLayerSize = 128;
    private double learningRate = 0.01;
    private double dropOut = 0.5;
    private int iterations = 1;
    private int batchSize = 16; // number of Examples until Sample/Test
    private int numEpochs = 1;
    
    private boolean enabletrainingUI = false;
    
    public Builder() {
      tagger = new SectorTagger();
      ann = new SectorAnnotator(tagger);
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
    
    public Builder withLossFunction(LossFunctions.LossFunction lossFunc, Activation activation, boolean requireSubsampling) {
      this.lossFunc = lossFunc.getILossFunction();
      this.requireSubsampling = requireSubsampling;
      this.activation = activation;
      return this;
    }
    
    public Builder withLossFunction(ILossFunction lossFunc, Activation activation, boolean requireSubsampling) {
      this.lossFunc = lossFunc;
      this.requireSubsampling = requireSubsampling;
      this.activation = activation;
      return this;
    }
    
    public Builder withModelParams(int ffwLayerSize, int lstmLayerSize, int embeddingLayerSize) {
      this.ffwLayerSize = ffwLayerSize;
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
        
    public Builder withInputEncoders(String desc, Encoder bagEncoder, Encoder embEncoder, Encoder flagEncoder) {
      tagger.setInputEncoders(bagEncoder, embEncoder, flagEncoder);
      ann.getProvenance().setFeatures(desc);
      ann.addComponent(bagEncoder);
      ann.addComponent(embEncoder);
      ann.addComponent(flagEncoder);
      return this;
    }
    
    public Builder withTargetEncoder(Encoder targetEncoder) {
      tagger.setTargetEncoder(targetEncoder);
      ann.addComponent(targetEncoder);
      return this;
    }
    
    public Builder withExistingComponents(SectorAnnotator parent) {
      for(Map.Entry<String, AnnotatorComponent> comp : parent.components.entrySet()) {
        //if(!ann.components.containsKey(comp.getKey())) ann.addComponent(comp.getValue()); // String Key match
        if(!ann.components.containsValue(comp.getValue())) ann.addComponent(comp.getValue()); // Instance match
      }
      return this;
    }
        
    public Builder enableTrainingUI(boolean enable) {
      this.enabletrainingUI = enable;
      return this;
    }
    
    /** pretrain encoders */
    public Builder pretrain(Dataset train) {
      for(Encoder e : encoders) {
        e.trainModel(train.streamDocuments());
      }
      return this;
    }
    
    public SectorAnnotator build() {
      tagger.buildMultiFwBwSectorNetwork(ffwLayerSize, lstmLayerSize, embeddingLayerSize, iterations, learningRate, dropOut, lossFunc, activation);
      if(enabletrainingUI) tagger.enableTrainingUI();
      tagger.setRequireSubsampling(requireSubsampling);
      tagger.setTrainingParams(examplesPerEpoch, batchSize, numEpochs, true);
      ann.getProvenance().setTask(tagger.getId());
      tagger.setName(ann.getProvenance().toString());
      tagger.appendTrainLog(printParams());
      return ann;
    }
    
    private String printParams() {
      StringBuilder line = new StringBuilder();
      line.append("TRAINING PARAMS: ").append(tagger.getName()).append("\n");
      line.append("\nInput Encoders:\n");
      for(Encoder e : tagger.getEncoders()) {
        line.append(e.getId()).append("\t").append(e.getClass().getSimpleName()).append("\t").append(e.getVectorSize()).append("\n");
      }
      line.append("\nTarget Encoders:\n");
      for(Encoder e : tagger.getTargetEncoders()) {
        line.append(e.getId()).append("\t").append(e.getClass().getSimpleName()).append("\t").append(e.getVectorSize()).append("\n");
      }
      line.append("\nNetwork Params:\n");
      line.append("FF").append("\t").append(ffwLayerSize).append("\n");
      line.append("BLSTM").append("\t").append(lstmLayerSize).append("\n");
      line.append("EMB").append("\t").append(embeddingLayerSize).append("\n");
      line.append("\nTraining Params:\n");
      line.append("examples per epoch").append("\t").append(examplesPerEpoch).append("\n");
      line.append("epochs").append("\t").append(numEpochs).append("\n");
      line.append("iterations").append("\t").append(iterations).append("\n");
      line.append("batch size").append("\t").append(batchSize).append("\n");
      line.append("learning rate").append("\t").append(learningRate).append("\n");
      line.append("dropout").append("\t").append(dropOut).append("\n");
      line.append("loss").append("\t").append(lossFunc.toString()).append(requireSubsampling ? " (multi-class)" : " (single-class)").append("\n");
      line.append("\n");
      //System.out.println(line.toString());
      return line.toString();
    }

  }
  
}
