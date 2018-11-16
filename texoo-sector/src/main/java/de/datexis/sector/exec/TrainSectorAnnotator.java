package de.datexis.sector.exec;

import de.datexis.common.*;
import de.datexis.encoder.*;
import de.datexis.encoder.impl.*;
import de.datexis.model.*;
import de.datexis.parvec.encoder.ParVecWordsEncoder;
import de.datexis.rnn.loss.MultiClassDosSantosPairwiseRankingLoss;
import de.datexis.sector.SectorAnnotator;
import de.datexis.sector.encoder.*;
import de.datexis.sector.model.SectionAnnotation;
import de.datexis.sector.reader.WikiSectionReader;
import java.io.IOException;
import java.util.ArrayList;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.deeplearning4j.ui.api.UIServer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main Controller for training of SECTOR models.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class TrainSectorAnnotator {

  protected final static Logger log = LoggerFactory.getLogger(TrainSectorAnnotator.class);

  public static void main(String[] args) throws IOException {
    
    final ExecParams params = new ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
      new TrainSectorAnnotator().runTraining(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("texoo-train-sector", "TeXoo: train SectorAnnotator from WikiSection dataset", params.setUpCliOptions(), "", true);
      System.exit(1);
    }
   
  }
  
  protected static class ExecParams implements CommandLineParser.Options {

    protected String inputFile;
    protected String outputPath = null;
    protected String embeddingsFile = null;
    protected boolean trainingUI = false;

    @Override
    public void setParams(CommandLine parse) {
      inputFile = parse.getOptionValue("i");
      outputPath = parse.getOptionValue("o");
      embeddingsFile = parse.getOptionValue("e");
      trainingUI = parse.hasOption("u");
    }

    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "input", true, "file name of WikiSection training dataset");
      op.addRequiredOption("o", "output", true, "path to create and store the model");
      op.addOption("e", "embedding", true, "path to word embedding model");
      op.addOption("u", "ui", false, "enable training UI (http://127.0.0.1:9000)");
      return op;
    }

  }
  
  protected void runTraining(ExecParams params) throws IOException {
    
    // Configure parameters
    Resource trainingPath = Resource.fromDirectory(params.inputFile);
    //Resource validationPath = Resource.fromDirectory(params.validationPath);
    //Resource testPath = Resource.fromDirectory(params.testPath);
    Resource output = Resource.fromDirectory(params.outputPath);
    
    // Read datasets
    Dataset train = WikiSectionReader.readDatasetFromJSON(trainingPath);
    WordHelpers.Language lang = train.getName().contains("de_") ? WordHelpers.Language.DE : WordHelpers.Language.EN;
    //Dataset validation = CoNLLDatasetReader.readDataset(validationPath, validationPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);
    //Dataset test = CoNLLDatasetReader.readDataset(testPath, testPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);

    // Configure encoders and model
    LookupCacheEncoder targetEncoder = initializeClassLabelsTarget(train, lang);
    //LookupCacheEncoder targetEncoder = initializeHeadingsTask(train, lang);
    SectorAnnotator.Builder builder;
    if(params.embeddingsFile != null) {
      builder = initializeClassLabelsModel_wemb(train, lang, Resource.fromFile(params.embeddingsFile));
    } else {
      builder = initializeClassLabelsModel_bloom(train, lang);
    }
    SectorAnnotator sector = builder.withDataset(train.getName(), lang)
                                    .withTargetEncoder(targetEncoder)
                                    .enableTrainingUI(params.trainingUI)
                                    .build();
    
    try {
      // Train model
      //sector.trainModelEarlyStopping(train, validation, 10, 6, 20);
      sector.trainModel(train, 6);
      // Save model
      output.toFile().mkdirs();
      sector.writeModel(output, sector.getTagger().getName());
      sector.writeTrainLog(output);
      sector.writeTestLog(output);
    } finally {
      if(params.trainingUI) UIServer.getInstance().stop();
    }
    
  }
  
  /**
   * Initialize target encoder for the class labels task (single-label classification)
   */
  protected LookupCacheEncoder initializeClassLabelsTarget(Dataset train, WordHelpers.Language lang) {
    
    // preprocess Section Annotations
    ArrayList<String> sections = new ArrayList<>();
      for(Document doc : train.getDocuments()) {
        for(SectionAnnotation ann : doc.getAnnotations(SectionAnnotation.class)) {
          sections.add(ann.getSectionLabel());
        }
      }
    
    // build Section Encoder
    ClassEncoder targetEncoder = new ClassEncoder();
    targetEncoder.trainModel(sections, 0);
    ClassTag.Factory classTags = new ClassTag.Factory(targetEncoder);
    for(Document doc : train.getDocuments()) classTags.attachFromSectionAnnotations(doc, Annotation.Source.GOLD);
      
    return targetEncoder;
    
  }
  
  /**
   * Initialize target encoder for the headings prediction task (multi-label classification)
   */
  protected LookupCacheEncoder initializeHeadingsTarget(Dataset train, WordHelpers.Language lang) {
    
    // preprocess Section Annotations
    ArrayList<String> headings = new ArrayList<>();
    for(Document doc : train.getDocuments()) {
      for(SectionAnnotation ann : doc.getAnnotations(SectionAnnotation.class)) {
        headings.add(ann.getSectionHeading());
      }
    }
    
    // build Headline Encoder
    HeadingEncoder targetEncoder = new HeadingEncoder();
    targetEncoder.trainModel(headings, 3, lang);
    HeadingTag.Factory headingTags = new HeadingTag.Factory(targetEncoder);
    for(Document doc : train.getDocuments()) headingTags.attachFromSectionAnnotations(doc, Annotation.Source.GOLD);
    
    return targetEncoder;
    
  }
  
  protected SectorAnnotator.Builder initializeClassLabelsModel_bloom(Dataset train, WordHelpers.Language lang) {
    
    BloomEncoder bloom = new BloomEncoder(4096, 5);
    bloom.trainModel(train.getDocuments(), 5, lang);
    StructureEncoder structure = new StructureEncoder();
    for(Document doc : train.getDocuments()) structure.encodeEach(doc, Sentence.class);
      
    return new SectorAnnotator.Builder()
        .withId("SECTOR_class")
        .withInputEncoders("bloom+mxcent+128", bloom, new DummyEncoder(), structure)
        .withLossFunction(new LossMCXENT(), Activation.SOFTMAX, false)
        .withModelParams(0, 256, 128)
        .withTrainingParams(0.01, 0.5, 2048, 16, 1);
  }
  
  protected SectorAnnotator.Builder initializeHeadingsModel_bloom(Dataset train, WordHelpers.Language lang) {
    
    BloomEncoder bloom = new BloomEncoder(4096, 5);
    bloom.trainModel(train.getDocuments(), 5, lang);
    StructureEncoder structure = new StructureEncoder();
    for(Document doc : train.getDocuments()) structure.encodeEach(doc, Sentence.class);
    
    return new SectorAnnotator.Builder()
        .withId("SECTOR_head")
        .withInputEncoders("bloom+rank+128", bloom, new DummyEncoder(), structure)
        .withLossFunction(new MultiClassDosSantosPairwiseRankingLoss(), Activation.IDENTITY, false)
        .withModelParams(0, 256, 128)
        .withTrainingParams(0.001, 0.0, 2048, 16, 1);
  }
  
  protected SectorAnnotator.Builder initializeClassLabelsModel_wemb(Dataset train, WordHelpers.Language lang, Resource embeddingModel) throws IOException {
    
    ParVecWordsEncoder wordEmb = new ParVecWordsEncoder();
    wordEmb.loadModel(embeddingModel);
    StructureEncoder structure = new StructureEncoder();
    for(Document doc : train.getDocuments()) structure.encodeEach(doc, Sentence.class);
    
    return new SectorAnnotator.Builder()
        .withId("SECTOR_class")
        .withInputEncoders("wemb+mxcent+128", new DummyEncoder(), wordEmb, structure)
        .withLossFunction(new LossMCXENT(), Activation.SOFTMAX, false)
        .withModelParams(0, 256, 128)
        .withTrainingParams(0.01, 0.5, 2048, 16, 1);
  }
  
}
