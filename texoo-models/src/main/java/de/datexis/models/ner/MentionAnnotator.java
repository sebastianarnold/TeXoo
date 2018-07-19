package de.datexis.models.ner;

import de.datexis.annotator.Annotator;
import de.datexis.common.Resource;
import de.datexis.common.Timer;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderSet;
import de.datexis.tagger.Tagger;
import de.datexis.model.Document;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Sentence;
import de.datexis.model.tag.BIOESTag;
import de.datexis.models.ner.eval.HTMLExport;
import de.datexis.models.ner.eval.MentionAnnotatorEvaluation;
import de.datexis.models.ner.tagger.MentionTagger;
import java.io.IOException;
import java.util.Collection;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple NER Annotator that returns a Dataset with Named Entity mentions
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MentionAnnotator extends Annotator {

  protected final static Logger log = LoggerFactory.getLogger(MentionAnnotator.class);
  protected Resource bestModel;
  
  public MentionAnnotator() {
  }
  
  public MentionAnnotator(Tagger root) {
    super(root);
  }
  
  @Override
  public MentionTagger getTagger() {
    return (MentionTagger) tagger;
  }
  
  @Override
  public String toString() {
    return getProvenance().toString();
  }
  
  @Override
  public void annotate(Collection<Document> docs) {
    //timer.start();
    // 1. Encode Tokens using an Iterator with Encoders and Tagset
    //Document2MentionIterator it = new Document2MentionIterator(docs, "", tagger.getEncoders(), tagger.getTagset(), -1, 10, false);
    // 2. Tag Documents using a Tagger
    getTagger().tag(docs);
    
    // 3. Annotate documents using an Annotator
    createAnnotations(docs, Annotation.Source.PRED);
    //timer.stop();
    //log.debug("Annotated " + data.countSentences() + " sentences, " + data.countAnnotations(Annotation.Source.PRED) + " mentions [" + timer.get() + " total]");
  }
  
  public void trainModel(Dataset train, Dataset test, WordHelpers.Language lang) {
    
    provenance.setDataset(train.getName());
    provenance.setLanguage(lang.toString().toLowerCase());
    getTagger().setName(provenance.toString());
    
    // train tagger
    createTags(train.getDocuments(), Annotation.Source.GOLD);
    getTagger().trainModel(train, Annotation.Source.GOLD);

    // test tagger
    createTags(test.getDocuments(), Annotation.Source.GOLD);
    getTagger().testModel(test, Annotation.Source.GOLD);
    
  }
  
  public void trainModel(Dataset train, Annotation.Source annotationSource, WordHelpers.Language lang) {
    trainModel(train, annotationSource, lang, -1, true, true);
  }
  
  /**
   * Trains the NER model using a complete Dataset with MentionAnnotations
   * @param train - dataset do use for training
   * @param annotationSource - use MentionAnnotations of given source to create tags
   * @param lang - language used for preprocessing
   * @param limitExamples - use only n examples for training (after randomization), -1 for all examples
   * @param incremental - don't reset the model before training
   * @param randomize - randomize examples after each epoch
   */
  public void trainModel(Dataset train, Annotation.Source annotationSource, WordHelpers.Language lang, int limitExamples, boolean incremental, boolean randomize) {
    
    provenance.setDataset(train.getName());
    provenance.setLanguage(lang.toString().toLowerCase());
    getTagger().setName(provenance.toString());
    
    // train tagger
    createTags(train.getDocuments(), annotationSource);
    getTagger().trainModel(train, annotationSource, limitExamples, randomize);
    
  }
  
  public void trainModelEarlyStopping(Dataset train, Dataset validation, Annotation.Source annotationSource, WordHelpers.Language lang, int epochSize, int minEpochs, int maxEpochs, int maxEpochsWithNoImprovement) {
    
    provenance.setDataset(train.getName());
    provenance.setLanguage(lang.toString().toLowerCase());
    getTagger().setName(provenance.toString());
    
    // train tagger
    createTags(train.getDocuments(), annotationSource);
    
    Timer timer = new Timer();
    
    int epoch = 1;
    double score = 0;
    double bestScore = 0;
    int retries = maxEpochsWithNoImprovement;
    
    timer.start();
    do {
      getTagger().appendTrainLog("\n");
      getTagger().appendTrainLog("EPOCH " + epoch + ": training " + tagger.getName());
      getTagger().trainModel(train, annotationSource, epochSize, true);
      // test NER model
      getTagger().appendTestLog("Testing epoch " + epoch);
      annotate(validation.getDocuments());
      MentionAnnotatorEvaluation eval = new MentionAnnotatorEvaluation("TraiNER epoch " + epoch, annotationSource, Annotation.Source.PRED, Annotation.Match.STRONG);
      eval.calculateScores(validation.getDocuments());
      eval.printAnnotationStats();
      score = eval.getScore();
      timer.setSplit("epoch");
      getTagger().appendTrainLog("EPOCH " + epoch + " complete: score " + score, timer.getLong("epoch"));
      if(score >= bestScore) {
        bestModel = Resource.createTempDirectory();
        writeModel(bestModel, getTagger().getName());
        try {
          HTMLExport htmlTest = new HTMLExport(validation.getDocuments(), BIOESTag.class, annotationSource, Annotation.Source.PRED);
          FileUtils.writeStringToFile(bestModel.resolve("test_" + epoch + ".html").toFile(), htmlTest.getHTML());
        } catch (IOException ex) {
          log.error("Could not write output: " + ex.toString());
        }
        bestScore = score;
        retries = maxEpochsWithNoImprovement;
      } else {
        retries--;
      }
      epoch++;
    } while((epoch <= minEpochs || retries >= 0) && !(epoch > maxEpochs));
    timer.stop();
    
    getTagger().appendTrainLog("Training complete: " + tagger.getName() + " with score " + bestScore, timer.getLong());
    getTagger().appendTrainLog("\n");
    
  }
  
  /**
   * Writes <name>.xml and binary models
   * @param path Directory to write to
   */
  public void writeBestModel(Resource path, String name) throws IOException {
    FileUtils.copyDirectory(bestModel.toFile(), path.toFile());
  }
  
  
  /**
   * Trains the NER model using selected Sentences (BIO2Tags are required and will not be generated)
   * @param sentences
   * @param tagSource
   * @param lang 
   */
  public void trainModel(Collection<Sentence> sentences, Annotation.Source tagSource, WordHelpers.Language lang) {
    
    //provenance.setDataset(train.getName());
    provenance.setLanguage(lang.toString().toLowerCase());
    getTagger().setName(provenance.toString());
    
    // train tagger
    //createTags(train.getDocuments(), tagSource);
    getTagger().trainModel(sentences, tagSource, true);
    
  }
  
  protected void createTags(Iterable<Document> docs, Annotation.Source expected) {
    for(Document doc : docs) {
      if(!doc.isTagAvaliable(expected, BIOESTag.class) && doc.isTagAvaliable(expected, BIO2Tag.class)) {
        BIO2Tag.convertToBIOES(doc, expected);
        doc.setTagAvailable(expected, BIOESTag.class, true);
      } else if(!doc.isTagAvaliable(expected, BIOESTag.class)) {
        MentionAnnotation.createTagsFromAnnotations(doc, expected, BIOESTag.class);
        doc.setTagAvailable(expected, BIOESTag.class, true);
      }
    }
  }
  
  protected void createAnnotations(Iterable<Document> docs, Annotation.Source expected) {
    for(Document doc : docs) {
      doc.clearAnnotations(expected, MentionAnnotation.class);
      if(doc.isTagAvaliable(expected, BIO2Tag.class)) {
        MentionAnnotation.annotateFromTags(doc, Annotation.Source.PRED, BIO2Tag.class);
      } else {
        log.error("BIO2Tag not set");
      }
    }
  }

  public static class Builder {
    
    MentionAnnotator ann;
    MentionTagger tagger;
    
    protected String types = MentionAnnotation.Type.GENERIC;
    protected Class tagset = BIOESTag.class;
    protected Encoder[] encoders = new Encoder[0];
    
    private int trainingSize = -1;
    private int ffwLayerSize = 300;
    private int lstmLayerSize = 100;
    private double learningRate = 0.001;
    private int iterations = 1;
    private int batchSize = 16; // number of Examples until Sample/Test
    private int numEpochs = 1;
    private int workers = 1;
    
    private boolean enabletrainingUI = false;
    
    public Builder() {
      tagger = new MentionTagger();
      ann = new MentionAnnotator(tagger);
    }
    
    public Builder withModelParams(int ffwLayerSize, int lstmLayerSize) {
      this.ffwLayerSize = ffwLayerSize;
      this.lstmLayerSize = lstmLayerSize;
      return this;
    }
        
     public Builder withTrainingParams(double learningRate, int batchSize, int numEpochs) {
       this.learningRate = learningRate;
       this.batchSize = batchSize;
       this.numEpochs = numEpochs;
       return this;
     }
     
     public Builder withWorkspaceParams(int workers) {
       this.workers = workers;
       return this;
     }
    
    public Builder withTypes(MentionAnnotation.Type types) {
      this.types = types.toString();
      return this;
    }
    
    public Builder withTypes(String types) {
      this.types = types;
      return this;
    }
    
    public Builder withEncoders(String desc, Encoder... encoders) {
      ann.getProvenance().setFeatures(desc);
      withEncoders(encoders);
      return this;
    }
    
    public Builder withEncoders(Encoder... encoders) {
      this.encoders = encoders;
      ann.getProvenance().setArchitecture(this.encoders.toString());
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
    
    public MentionAnnotator build() {
      for(Encoder e : encoders) {
        if(!e.isModelAvailable()) throw new IllegalArgumentException("encoder " + e.getId() + " has no model available, please consider pretrain()");
        ann.addComponent(e);
      }
      tagger.setTagset(tagset, types);
      tagger.setEncoders(new EncoderSet(encoders));
      tagger.build(ffwLayerSize, lstmLayerSize, iterations, learningRate * batchSize);
      if(enabletrainingUI) tagger.enableTrainingUI();
      tagger.setTrainingParams(batchSize, numEpochs, true);
      tagger.setWorkspaceParams(workers);
      ann.getProvenance().setTask("NER-" + types);
      tagger.setName(ann.getProvenance().toString());
      return ann;
    }

  }
  
}
