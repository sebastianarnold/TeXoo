package de.datexis.cdv.train;

import de.datexis.annotator.Annotator;
import de.datexis.annotator.AnnotatorFactory;
import de.datexis.cdv.CDVAnnotator;
import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.index.EntityIndex;
import de.datexis.cdv.loss.LossHuber;
import de.datexis.common.*;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderAnnotator;
import de.datexis.encoder.IEncoder;
import de.datexis.encoder.impl.BagOfWordsEncoder;
import de.datexis.encoder.impl.FastTextEncoder;
import de.datexis.encoder.impl.StructureEncoder;
import de.datexis.encoder.impl.Word2VecEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.parvec.encoder.ParVecWordsEncoder;
import de.datexis.preprocess.IdentityPreprocessor;
import de.datexis.retrieval.encoder.LSTMSentenceAnnotator;
import de.datexis.sector.encoder.ParVecSentenceEncoder;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;

/**
 * WWW2020: Training for CDV model
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class TrainCDVAnnotator {

  protected final static Logger log = LoggerFactory.getLogger(TrainCDVAnnotator.class);

  protected WordHelpers.Language lang;
  // input encoders
  protected Encoder sentenceEmb;
  protected StructureEncoder positionalEmb; // positional embedding
  // target decoders
  protected IEncoder entityEmb;
  protected IEncoder aspectEmb;
  // target indexes (pre-trained vectors)
  protected EntityIndex entityIndex = null;
  protected AspectIndex aspectIndex = null;
  
  public static void main(String[] args) throws IOException, ParseException {
    final TrainingParams params = new TrainingParams();
    final CommandLineParser parser = new CommandLineParser(params);
    try {
      parser.parse(args);
      new TrainCDVAnnotator().trainCompleteCDVModel(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("train-cdv", "TeXoo: train contextualized discourse vectors (CDV)", params.setUpCliOptions(), "", true);
      System.exit(1);
    } catch(Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
  }
  
  protected static class TrainingParams implements CommandLineParser.Options {

    protected String inputPath = null;
    protected String datasetName = "wd_disease";
    protected String inputEmbedding = null;
    protected String entityEmbedding = null;
    protected String aspectEmbedding = null;
    protected String searchPath = "models/common";
    protected String outputPath = "models";
    protected String modelName = null;
    protected boolean trainingUI = true;
    protected boolean entityModel = true;
    protected boolean aspectModel = true;
    protected boolean balancing = false;
    protected int epochs = 50;

    @Override
    public void setParams(CommandLine parse) {
      inputPath = parse.getOptionValue("i");
      datasetName = parse.getOptionValue("d");
      modelName = parse.getOptionValue("m");
      inputEmbedding = parse.getOptionValue("w");
      entityModel = parse.hasOption("e");
      entityEmbedding = parse.getOptionValue("e");
      aspectModel = parse.hasOption("a");
      aspectEmbedding = parse.getOptionValue("a");
      searchPath = parse.getOptionValue("s");
      balancing = parse.hasOption("b");
      trainingUI = parse.hasOption("u");
      outputPath = parse.getOptionValue("o", Configuration.getProperty("de.datexis.path.results"));
    }

    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "dataset", true, "path to the WikiSection training dataset");
      op.addRequiredOption("d", "datasetname", true, "name of the data set, e.g. en_disease");
      op.addRequiredOption("m", "modelname", true, "model name");
      op.addOption("w", "wordemb", true, "path to a pretrained embedding");
      op.addOption("e", "entity", true, "path to the entity embedding");
      op.addOption("a", "aspect", true, "path to the aspect embedding");
      op.addOption("o", "output", true, "path to create the output folder in");
      op.addOption("s", "search", true, "search path for pre-trained word embeddings");
      op.addOption("b", "balancing", false, "use class balancing during training");
      op.addOption("u", "ui", false, "enable training UI");
      return op;
    }

  }
  
  public void trainCompleteCDVModel(TrainingParams params) throws IOException {
  
    Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
    
    Resource path = Resource.fromDirectory(params.inputPath);
    Resource output = Resource.fromDirectory(params.outputPath);
    
    lang = params.datasetName.startsWith("de_") ? WordHelpers.Language.DE : WordHelpers.Language.EN;
  
    Dataset train = readDatasetFromJSON(path);
    log.info("read {} articles", train.countDocuments());
    
    loadSentenceEmbedding(params, train);
    if(params.aspectModel) loadAspectEmbedding(params, train);
    if(params.entityModel) loadEntityEmbedding(params, train);
    
    // --- build model ---------------------------------------------------------
    CDVAnnotator cdv;
    if(params.entityModel && params.aspectModel) {
      cdv = new CDVAnnotator.Builder()
        .withId("CDV-EA")
        .withClassBalancing(params.balancing)
        .withInputEncoders(params.modelName, sentenceEmb, positionalEmb)
        .withEntityAspectEncoders(entityIndex, aspectIndex)
        .withLossFunction(new LossHuber(), Activation.TANH)
        .withModelParams(768, 1024, 512)
        .withTrainingParams(0.001, 0.00, 0.0001, 16, params.epochs)
        .withDatasetLimit(-1, 396, 96) // wikipedia goes easily go up to 396x128
        .withDataset(params.datasetName, lang)
        .enableTrainingUI(params.trainingUI)
        .build();
    } else if(params.entityModel) {
      cdv = new CDVAnnotator.Builder()
        .withId("CDV-E")
        .withClassBalancing(params.balancing)
        .withInputEncoders(params.modelName, sentenceEmb, positionalEmb)
        .withEntityEncoder(entityIndex)
        .withLossFunction(new LossHuber(), Activation.TANH)
        .withModelParams(128, 512, 128)
        .withTrainingParams(0.0005, 0.0, 0.0001, 16, params.epochs)
        .withDatasetLimit(-1, 396, 96)
        .withDataset(params.datasetName, lang)
        .enableTrainingUI(params.trainingUI)
        .build();
    } else if(params.aspectModel) {
      cdv = new CDVAnnotator.Builder()
        .withId("CDV-A")
        .withClassBalancing(params.balancing)
        .withInputEncoders(params.modelName, sentenceEmb, positionalEmb)
        .withAspectEncoder(aspectIndex)
        .withLossFunction(new LossHuber(), Activation.TANH)
        .withModelParams(128, 512, 128)
        .withTrainingParams(0.0005, 0.0, 0.0001, 16, params.epochs)
        .withDatasetLimit(-1, 396, 96)
        .withDataset(params.datasetName, lang)
        .enableTrainingUI(params.trainingUI)
        .build();
    } else {
      throw new IllegalArgumentException("No entity or aspect index given.");
    }
  
    boolean success = false;
    try {
      positionalEmb.setCachingEnabled(true);
      sentenceEmb.setCachingEnabled(true);
      cdv.trainModel(train);
      String date = new SimpleDateFormat("yyMMdd_HHmm_").format(new Date());
      saveModel(cdv, output.resolve(date + cdv.getTagger().getName()));
      success = true;
      
    } catch(Throwable e) {
      e.printStackTrace();
    } finally {
      //if(params.trainingUI) UIServer.getInstance().stop();
      System.exit(success ? 0 : 1);
    }

  }
  
  private void loadSentenceEmbedding(TrainingParams params, Dataset train) throws IOException {
    Resource inputWordModel = params.inputEmbedding == null ? null : Resource.fromFile(params.inputEmbedding);
    if(inputWordModel != null) {
      if(params.inputEmbedding.endsWith("pv.zip")) {
        sentenceEmb = loadParVecEmbedding(inputWordModel);
      } else if(params.inputEmbedding.endsWith(".bin") ||
        params.inputEmbedding.endsWith(".bin.gz")) {
        sentenceEmb = loadFastTextEmbedding(inputWordModel);
      } else if(params.inputEmbedding.equalsIgnoreCase("ELMo")) {
        throw new UnsupportedOperationException("REST embedding is not possible outside Beuth infrastructure");
      } else if(params.inputEmbedding.equalsIgnoreCase("BERT-base")) {
        throw new UnsupportedOperationException("REST embedding is not possible outside Beuth infrastructure");
      } else if(params.inputEmbedding.equalsIgnoreCase("BioBERT")) {
        throw new UnsupportedOperationException("REST embedding is not possible outside Beuth infrastructure");
      } else if(params.inputEmbedding.equalsIgnoreCase("BERT-large")) {
        throw new UnsupportedOperationException("REST embedding is not possible outside Beuth infrastructure");
      } else {
        sentenceEmb = loadWord2VecEmbedding(inputWordModel);
      }
    } else {
      sentenceEmb = new BagOfWordsEncoder(); // use BOW word embedding
      sentenceEmb.trainModel(train.getDocuments());
    }
    positionalEmb = new StructureEncoder();
  }
  
  private void loadAspectEmbedding(TrainingParams params, Dataset train) throws IOException {
    Resource aspectModel = Resource.fromFile(params.aspectEmbedding);
    Resource searchPath = Resource.fromDirectory(params.searchPath);
    Resource indexPath = aspectModel.resolve("aspect.index.bin");
    if(aspectModel.isDirectory()) {
      Annotator ann = AnnotatorFactory.loadAnnotator(aspectModel, searchPath);
      if(ann instanceof EncoderAnnotator) {
        EncoderAnnotator enc = (EncoderAnnotator) ann;
        aspectEmb = enc.getEncoder(); // TODO: we might reuse the input embedding if they are the same
        if(enc.getEncoder() instanceof FastTextEncoder)
          ((FastTextEncoder) enc.getEncoder()).setModelAsReference(); // save pre-trained models as reference only
      } else if(ann instanceof LSTMSentenceAnnotator) {
        LSTMSentenceAnnotator lstm = (LSTMSentenceAnnotator) ann;
        if(lstm.getTagger().getInputEncoder() instanceof FastTextEncoder)
          ((FastTextEncoder) lstm.getTagger().getInputEncoder()).setModelAsReference(); // save pre-trained models as reference only
        aspectEmb = lstm.asEncoder();
      }
    } else if(params.aspectEmbedding.endsWith(".bin") ||
      params.aspectEmbedding.endsWith(".bin.gz")) {
      aspectEmb = loadFastTextEmbedding(aspectModel);
    } else {
      aspectEmb = loadWord2VecEmbedding(aspectModel);
    }
    // Load pre-trained index from encoder
    aspectIndex = new AspectIndex(aspectEmb);
    if(indexPath.exists()) aspectIndex.loadModel(indexPath);
    else throw new IllegalArgumentException("Aspect encoder needs to provide a knowledge base called aspect.index.bin");
    //else aspectIndex.trainModel(train.getDocuments());
  }
  
  private void loadEntityEmbedding(TrainingParams params, Dataset train) throws IOException {
    Resource entityModel = Resource.fromFile(params.entityEmbedding);
    Resource searchPath = Resource.fromDirectory(params.searchPath);
    Resource indexPath = entityModel.resolve("entity.index.bin");
    if(entityModel.isDirectory()) {
      Annotator ann = AnnotatorFactory.loadAnnotator(entityModel, searchPath);
      if(ann instanceof EncoderAnnotator) {
        EncoderAnnotator enc = (EncoderAnnotator) ann;
        entityEmb = enc.getEncoder(); // TODO: we might reuse the input embedding if they are the same
        if(enc.getEncoder() instanceof FastTextEncoder)
          ((FastTextEncoder) enc.getEncoder()).setModelAsReference(); // save pre-trained models as reference only
      } else if(ann instanceof LSTMSentenceAnnotator) {
        LSTMSentenceAnnotator lstm = (LSTMSentenceAnnotator) AnnotatorFactory.loadAnnotator(entityModel, searchPath);
        if(lstm.getTagger().getInputEncoder() instanceof FastTextEncoder)
          ((FastTextEncoder) lstm.getTagger().getInputEncoder()).setModelAsReference(); // save pre-trained models as reference only
        entityEmb = lstm.asEncoder();
      }
    } else if(params.entityEmbedding.endsWith(".bin") ||
      params.entityEmbedding.endsWith(".bin.gz")) {
      entityEmb = loadFastTextEmbedding(entityModel);
    } else {
      entityEmb = loadWord2VecEmbedding(entityModel);
    }
    // Load pre-trained Knowledge Base or build a new one
    entityIndex = new EntityIndex(entityEmb);
    if(indexPath.exists()) entityIndex.loadModel(indexPath);
    else throw new IllegalArgumentException("Entity encoder needs to provide a knowledge base called entity.index.bin");
  }
  
  private ParVecWordsEncoder loadParVecEmbedding(Resource parvecModel) throws IOException {
    ParVecSentenceEncoder parEmb = new ParVecSentenceEncoder();
    parEmb.loadModel(parvecModel);
    ParVecWordsEncoder wordEmb = new ParVecWordsEncoder();
    wordEmb.loadModel(parvecModel);
    return wordEmb;
  }
  
  private Word2VecEncoder loadWord2VecEmbedding(Resource wordEmbedding) throws IOException {
    Word2VecEncoder word2vec = new Word2VecEncoder();
    word2vec.loadModelAsReference(wordEmbedding);
    word2vec.setPreprocessor(new IdentityPreprocessor()); // for GloVe
    return word2vec;
  }
  
  private FastTextEncoder loadFastTextEmbedding(Resource wordEmbedding) throws IOException {
    FastTextEncoder sentenceEmbedding = new FastTextEncoder();
    sentenceEmbedding.loadModelAsReference(wordEmbedding);
    return sentenceEmbedding;
  }
  
  private void saveModel(Annotator annotator, Resource outputPath) throws IOException {
    outputPath.toFile().mkdirs();
    annotator.writeModel(outputPath);
    annotator.writeTrainLog(outputPath);
    annotator.writeTestLog(outputPath);
    log.info("model written to {}", outputPath.toString());
  }
  
  public static Dataset readDatasetFromJSON(Resource path) throws IOException {
    log.info("Reading Wiki Articles from {}", path.toString());
    Dataset result = new Dataset(path.getFileName().replace(".json", ""));
    Iterator<Document> it = ObjectSerializer.readJSONDocumentIterable(path);
    while(it.hasNext()) {
      Document doc = it.next();
      for(Annotation ann : doc.getAnnotations()) {
        ann.setSource(Annotation.Source.GOLD);
        ann.setConfidence(1.0);
      }
      result.addDocument(doc);
    }
    return result;
  }
  
}
