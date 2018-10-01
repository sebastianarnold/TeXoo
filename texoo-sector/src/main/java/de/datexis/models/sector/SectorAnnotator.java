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
import de.datexis.models.sector.tagger.SectorEncoder;
import de.datexis.models.sector.tagger.SectorTagger;
import java.util.Collection;
import java.util.Map;
import java.util.stream.Collectors;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
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
    NONE, // don't segment, only tag sentences
    GOLD, // use provided gold standard segmentation (perfect case)
    NEWLINES, // segment at every newline (will produce too many segments)
    TARGET_LABEL, // segment if top-2 labels change
    TARGET_PCA, // segment if top-8 principal components change
    TARGET_MAGNITUDE, // segmentation based on edge detection on target magnitude
    EMBEDDING_MAGNITUDE, // segmentation based on edge detection on embedding magnitude
    FWBW_MAGNITUDE, // segmentation based on edge detection on FW/BW embedding magnitude
  };
  
  public SectorAnnotator() {
  }
  
  public SectorAnnotator(Tagger root) {
    super(root);
  }
  
  protected SectorAnnotator(AnnotatorComponent comp) {
    super(comp);
  }
  
  @Override
  public SectorTagger getTagger() {
    return (SectorTagger) super.getTagger();
  }
  
  public LookupCacheEncoder getTargetEncoder() {
    return (LookupCacheEncoder) getTagger().getTargetEncoder();
  }
  
  @Override
  public void annotate(Collection<Document> docs) {
    annotate(docs, SegmentationMethod.NEWLINES);
  }
  
  public void annotate(Collection<Document> docs, SegmentationMethod segmentation) {
    annotate(docs, segmentation, false);
  }
  
  public void annotate(Collection<Document> docs, SegmentationMethod segmentation, boolean alignFwBwEmbeddings) {
    // use tagger to generate and attach PRED vectors to Sentences
    log.info("Running SECTOR neural net encoding...");
    getTagger().attachVectors(docs, DocumentSentenceIterator.Stage.ENCODE, getTargetEncoder().getClass(), alignFwBwEmbeddings);
    if(!segmentation.equals(SegmentationMethod.NONE)) segment(docs, segmentation);
  }
  
  public void segment(Collection<Document> docs, SegmentationMethod segmentation) {
    // create Annotations and attach vectors
    log.info("Predicting segmentation {}...", segmentation.toString());
    detectSections(docs, segmentation);
    // attach vectors to annotations
    log.info("Attaching Annotations...");
    for(Document doc : docs) attachVectorsToAnnotations(doc, getTargetEncoder());
    log.info("Segmentation done.");
  }
  
  protected void detectSections(Collection<Document> docs, SegmentationMethod segmentation) {
    WorkspaceMode cMode = getTagger().getNN().getConfiguration().getInferenceWorkspaceMode();
    getTagger().getNN().getConfiguration().setTrainingWorkspaceMode(getTagger().getNN().getConfiguration().getInferenceWorkspaceMode());
    MemoryWorkspace workspace =
            getTagger().getNN().getConfiguration().getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                    : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread();
    
//    WorkspaceMode wsm = getTagger().getNN().getConfiguration().getInferenceWorkspaceMode();
//    LayerWorkspaceMgr mgr;
//    if(wsm == WorkspaceMode.NONE){
//        mgr = LayerWorkspaceMgr.noWorkspaces();
//    } else {
//        mgr = LayerWorkspaceMgr.builder()
//                .noWorkspaceFor(ArrayType.ACTIVATIONS)
//                .noWorkspaceFor(ArrayType.INPUT)
//                .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
//                .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
//                .build();
//    }
//    mgr.setHelperWorkspacePointers(helperWorkspaces);
    
    for(Document doc : docs) {
      try (MemoryWorkspace wsE = workspace.notifyScopeEntered()) {
        switch(segmentation) {
          case GOLD: {
            applySectionsFromGold(doc); 
          } break;
          case TARGET_LABEL: {
            applySectionsFromTargetLabels(doc, getTargetEncoder(), 2); 
          } break;
          case TARGET_PCA: {
            detectSectionsFromTargetPCA(doc, getTargetEncoder()); 
          } break;
          case TARGET_MAGNITUDE: {
            INDArray mag = detectSectionsFromTargetMagnitude(doc, getTargetEncoder()); 
            applySectionsFromEdges(doc, detectEdges(mag));
          } break;
          case EMBEDDING_MAGNITUDE: {
            INDArray mag = detectSectionsFromEmbeddingMagnitude(doc); 
            applySectionsFromEdges(doc, detectEdges(mag));
          } break;
          case FWBW_MAGNITUDE: {
            INDArray mag = detectSectionsFromFWBWEmbeddingMagnitude(doc); 
            applySectionsFromEdges(doc, detectEdges(mag));
          } break;
          case NEWLINES:
          default: {
            applySectionsFromNewlines(doc);
          }
        }
      }
    }
    
    getTagger().getNN().getConfiguration().setTrainingWorkspaceMode(cMode);
    
  }
  
  public double evaluateModel(Dataset test) {
    SectorEvaluation eval;
    if(getTargetEncoder().getClass() == HeadingEncoder.class) {
      HeadingEncoder headings = ((HeadingEncoder)getComponent(HeadingEncoder.ID));
      eval = new SectorEvaluation(test.getName(), Annotation.Source.GOLD, Annotation.Source.PRED, headings);
      // we need tags for sentence-level evaluation
      log.info("Creating tags...");
      removeTags(test.getDocuments(), Annotation.Source.PRED);
      createHeadingTags(test.getDocuments(), Annotation.Source.GOLD, headings);
      createHeadingTags(test.getDocuments(), Annotation.Source.PRED, headings);
    } else if(getTargetEncoder().getClass() == ClassEncoder.class) {
      ClassEncoder classes = ((ClassEncoder)getComponent(ClassEncoder.ID));
      eval = new SectorEvaluation(test.getName(), Annotation.Source.GOLD, Annotation.Source.PRED, classes);
      // we need tags for sentence-level evaluation
      removeTags(test.getDocuments(), Annotation.Source.PRED);
      createClassTags(test.getDocuments(), Annotation.Source.GOLD, classes);
      createClassTags(test.getDocuments(), Annotation.Source.PRED, classes);
    } else {
      throw new IllegalArgumentException("Target encoder has no evaluation: " + getTargetEncoder().getClass().toString());
    }
    // calculate and print scores
    eval.calculateScores(test);
    getTagger().appendTestLog(eval.printDatasetStats(test));
    getTagger().appendTestLog(eval.printEvaluationStats());
    getTagger().appendTestLog(eval.printSingleClassStats());
    return eval.getScore();
  }
  
  public void trainModel(Dataset train, int numEpochs) {
    // train tagger
    getTagger().trainModel(train, numEpochs);
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
  
  protected static void removeTags(Iterable<Document> docs, Annotation.Source source) {
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
        ann.setSectionHeading(targetEncoder.getNearestNeighbours(pred, 3).toString());
        ann.setConfidence(targetEncoder.getMaxConfidence(pred));
      }
    }
  }
  
  /**
   * Add PRED SectionAnnotations from provided gold standard segmentation (perfect case)
   */
  protected static void applySectionsFromGold(Document doc) {
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
  protected static void applySectionsFromNewlines(Document doc) {
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
  protected static void applySectionsFromTargetLabels(Document doc, LookupCacheEncoder targetEncoder, int k) {
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
  
  protected static void detectSectionsFromTargetPCA(Document doc, LookupCacheEncoder targetEncoder) {
    
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
  
  protected static INDArray detectSectionsFromTargetMagnitude(Document doc, LookupCacheEncoder targetEncoder) {
    
    int PCA_DIMS = 8;
    
    if(doc.countSentences() < 2) return null;
      
    // initialize target matrix
    INDArray docTargets = getLayerMatrix(doc, targetEncoder.getClass());
    
    INDArray docTargetPCA = pca(docTargets, PCA_DIMS);
    INDArray docTargetSmooth = gaussianSmooth(docTargetPCA);
    INDArray docTargetMag = magnitude(docTargetSmooth);
    
    return docTargetMag;
    
  }
    
  
  protected static void applySectionsFromEdges(Document doc, INDArray docEdges) {
    
    // no sentence
    if(doc.countSentences() < 1) {
      log.warn("Empty document");
      return;
    }
    
    // single sentence
    if(docEdges == null || doc.countSentences() < 2) {
      SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
      section.setBegin(doc.getBegin());
      section.setEnd(doc.getEnd());
      doc.addAnnotation(section);
      return;
    }
    
    // start first section
    int sectionLength = 0;
    SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
    section.setBegin(doc.getBegin());

    int t = 0;
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
   * Add PRED SectionAnnotations based on edge detection on FW/BW embedding magnitude.
   */
  protected static INDArray detectSectionsFromEmbeddingMagnitude(Document doc) {
    
    int PCA_DIMS = 16;
    
    if(doc.countSentences() < 2) return null;
    
    // initialize embedding matrix
    INDArray docEmbs = getEmbeddingMatrix(doc);
    
    INDArray docPCA = pca(docEmbs, PCA_DIMS);
    INDArray docSmooth = gaussianSmooth(docPCA);
    INDArray docMag = magnitude(docSmooth);
    
    return docMag;
    
  }
  
  /**
   * Add PRED SectionAnnotations based on edge detection on FW/BW embedding magnitude.
   */
  protected static INDArray detectSectionsFromFWBWEmbeddingMagnitude(Document doc) {
    
    int PCA_DIMS = 16;
    
    if(doc.countSentences() < 1) return null;
    Sentence sent = doc.getSentence(0);
      
    // initialize FW/BW matrices
    long layerSize = sent.getVector("embeddingFW").length();
    INDArray docFW = Nd4j.zeros(doc.countSentences(), layerSize);
    INDArray docBW = Nd4j.zeros(doc.countSentences(), layerSize);
    
    // fill FW/BW matrices
    int t = 0;
    for(Sentence s : doc.getSentences()) {
      docFW.getRow(t).assign(s.getVector("embeddingFW"));
      docBW.getRow(t).assign(s.getVector("embeddingBW"));
      t++;
    }
    
    INDArray docFwPCA = docFW.mmul(PCA.pca_factor(docFW.dup(), PCA_DIMS, true));
    INDArray docBwPCA = docBW.mmul(PCA.pca_factor(docBW.dup(), PCA_DIMS, true));
    INDArray docFwPCAs = gaussianSmooth(docFwPCA, 1.5);
    INDArray docBwPCAs = gaussianSmooth(docBwPCA, 1.5);
    INDArray docMag = magnitude(docFwPCAs, docBwPCAs);
    
    return docMag;
    
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
    return getLayerMatrix(doc, layerClass.getSimpleName());
  }
  
  /**
   * @return Matrix sentences x layersize that contains Sector embeddings for a Document
   */
  protected static INDArray getEmbeddingMatrix(Document doc) {
    return getLayerMatrix(doc, SectorEncoder.class);
  }
  
  protected static INDArray pca(INDArray m, int dimensions) {
    return m.mmul(PCA.pca_factor(m.dup(), dimensions, true));
  }
  
  protected static INDArray gaussianSmooth(INDArray target) {
    return gaussianSmooth(target, 2.5);
  }
  
  protected static INDArray gaussianSmooth(INDArray target, double sd) {
    INDArray matrix = target.dup('c');
    INDArray kernel = Nd4j.zeros(matrix.rows(), 1, 'c');
    //INDArray docTargetsSmooth = Convolution.convn(docTargetsPCA, kernel, Convolution.Type.FULL);
    INDArray smooth = Nd4j.zerosLike(target);
    // convolution
    for(int t=0; t<kernel.length(); t++) {
      NormalDistribution dist = new NormalDistribution(t, sd);
      for(int k=0; k<kernel.length(); k++) {
        kernel.putScalar(k, dist.density(k));
      }
      INDArray conv = matrix.mulColumnVector(kernel); // TODO: mul takes a long time
      smooth.getRow(t).assign(conv.sum(0)); // TODO: sum takes a long time
    }
    return smooth;
  }
  
  /**
   * Returns a matrix [Tx1] that contains cosine distances between forward and backward layer.
   */
  protected static INDArray magnitude(INDArray fw, INDArray bw) {
    INDArray mag = Nd4j.zeros(fw.rows(), 1);
    for(int t = 1; t < mag.rows() - 1; t++) {
      double fwd1 = Transforms.cosineDistance(fw.getRow(t), fw.getRow(t+1)); // first derivative - FW is too late
      double bwd1 = Transforms.cosineDistance(bw.getRow(t), bw.getRow(t-1)); // first derivative - BW is too early
      //mag.putScalar(t, 0, Math.sqrt(Math.pow(fwd1, 2) + Math.pow(bwd1, 2) / 2.)); // quadratic mean
      double geom = Math.sqrt(fwd1 * bwd1);
      mag.putScalar(t, 0, Double.isNaN(geom) ? 0. : geom); // geometric mean
    }
    return mag;
  }
  
  /**
   * Returns a matrix [Tx1] that contains cosine distances between t-1 and t.
   */
  protected static INDArray magnitude(INDArray target) {
    INDArray mag = Nd4j.zeros(target.rows(), 1);
    for(int t = 1; t < mag.rows(); t++) {
      mag.putScalar(t, 0, Transforms.cosineDistance(target.getRow(t), target.getRow(t-1))); // first derivative
    }
    return mag;
  }
  
  /**
   * Returns a matrix [Tx1] that contains edges in magnitude.
   */
  protected static INDArray detectEdges(INDArray mag) {
    if(mag == null) return null;
    INDArray result = Nd4j.zeros(mag.rows(), 1);
    for(int t = 1; t < result.rows() - 1; t++) {
      result.putScalar(t, 0, ((mag.getDouble(t - 1) < mag.getDouble(t)) && (mag.getDouble(t + 1) < mag.getDouble(t))) ? 1 : 0);
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
