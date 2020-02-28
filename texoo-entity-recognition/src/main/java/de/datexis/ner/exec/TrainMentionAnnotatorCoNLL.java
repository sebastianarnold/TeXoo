package de.datexis.ner.exec;

import com.google.common.primitives.Ints;
import de.datexis.common.CommandLineParser;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.common.WordHelpers.Language;
import de.datexis.encoder.impl.*;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.ner.MentionAnnotator;
import de.datexis.ner.eval.MentionAnnotatorEvaluation;
import de.datexis.ner.reader.CoNLLDatasetReader;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Optional;

/**
 * Main Controller for training of MentionAnnotator / NER models.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class TrainMentionAnnotatorCoNLL {

  protected final static Logger log = LoggerFactory.getLogger(TrainMentionAnnotatorCoNLL.class);

  public static void main(String[] args) throws IOException {
    
    final ExecParams params = new ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
      new TrainMentionAnnotatorCoNLL().runTraining(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("texoo-train-ner", "TeXoo: train MentionAnnotator with CoNLL annotations", params.setUpCliOptions(), "", true);
      System.exit(1);
    }
   
  }
  
  protected static class ExecParams implements CommandLineParser.Options {

    protected String trainingPath;
    protected String validationPath;
    protected String testPath;
    protected String outputPath;
    protected String embeddingsFile = null;
    protected String language;
    protected boolean trainingUI = false;
    protected int epochs = 10;
    protected int examples = -1;

    @Override
    public void setParams(CommandLine parse) {
      trainingPath = parse.getOptionValue("i");
      validationPath = parse.getOptionValue("v");
      testPath = parse.getOptionValue("t");
      outputPath = parse.getOptionValue("o");
      embeddingsFile = parse.getOptionValue("e");
      trainingUI = parse.hasOption("u");
      language = parse.getOptionValue("l", "en");
      epochs = Optional.ofNullable(Ints.tryParse(parse.getOptionValue("n", "10"))).orElse(10);
      examples = Optional.ofNullable(Ints.tryParse(parse.getOptionValue("m", "-1"))).orElse(10);
    }

    protected void TrainMentionAnnotatorCoNLL() {}
    
    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "input", true, "path to input training data (CoNLL format)");
      op.addRequiredOption("o", "output", true, "path to create and store the model");
      op.addOption("v", "validation", true, "path to validation data (CoNLL format)");
      op.addOption("t", "test", true, "path to test data (CoNLL format)");
      op.addOption("m", "examples", true, "limit number of examples per epoch (default: all)");
      op.addOption("n", "epochs", true, "number of epochs (default: 10)");
      op.addOption("e", "embedding", true, "path to word embedding model (default: letter-trigrams)");
      op.addOption("l", "language", true, "language to use for sentence splitting and stopwords (EN or DE)");
      op.addOption("u", "ui", false, "enable training UI (http://127.0.0.1:9000)");
      return op;
    }

  }
  
  protected void runTraining(ExecParams params) throws IOException {
    
    // Configure parameters
    Resource trainingPath = Resource.fromDirectory(params.trainingPath);
    Resource validationPath = Resource.fromDirectory(params.validationPath);
    Resource testPath = Resource.fromDirectory(params.testPath);
    Resource outputPath = Resource.fromDirectory(params.outputPath);
    Language lang = WordHelpers.getLanguage(params.language);
    
    // Read datasets
    Dataset train = CoNLLDatasetReader.readDataset(trainingPath, trainingPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);
    //Dataset validation = CoNLLDatasetReader.readDataset(validationPath, validationPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);
  
    // Initialize builder
    MentionAnnotator.Builder builder = new MentionAnnotator.Builder();
  
    // Configure input encoders (trigram, fasttext or word embeddings)
    Resource embeddingModel = Resource.fromFile(params.embeddingsFile);
    if(params.embeddingsFile == null) {
      TrigramEncoder trigram = new TrigramEncoder();
      trigram.trainModel(train.getDocuments(), 10);
      builder.withEncoders("tri", new PositionEncoder(), new SurfaceEncoder(), trigram);
    } else if(embeddingModel.getFileName().endsWith(".bin") || embeddingModel.getFileName().endsWith(".bin.gz")) {
      FastTextEncoder fasttext = new FastTextEncoder();
      fasttext.loadModel(embeddingModel);
      builder.withEncoders("ft", new PositionEncoder(), new SurfaceEncoder(), fasttext);
    } else {
      Word2VecEncoder word2vec = new Word2VecEncoder();
      word2vec.loadModel(embeddingModel);
      builder.withEncoders("emb", new PositionEncoder(), new SurfaceEncoder(), word2vec);
    }
  
    // Configure model parameters
    MentionAnnotator ner = builder
      .enableTrainingUI(params.trainingUI)
      .withTrainingParams(0.0001, 64, params.epochs)
      .withModelParams(512, 256)
      .withWorkspaceParams(1) // single worker
      .build();

    // Train model
    ner.trainModel(train, Annotation.Source.GOLD, lang, params.examples, false, true);
  
    // Save model
    System.out.println("saving model to path: " + outputPath);
    outputPath.toFile().mkdirs();
    ner.writeModel(outputPath);
    
    // Evaluate
    if(params.testPath != null) {
      Dataset test = CoNLLDatasetReader.readDataset(testPath, testPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);
      MentionAnnotatorEvaluation eval = new MentionAnnotatorEvaluation(testPath.getFileName(), Annotation.Match.STRONG);
      eval.calculateScores(test);
      eval.printAnnotationStats();
    }
    
  }
  
}
