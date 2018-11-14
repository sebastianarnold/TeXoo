package de.datexis.ner.exec;

import de.datexis.common.CommandLineParser;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.common.WordHelpers.Language;
import de.datexis.encoder.impl.PositionEncoder;
import de.datexis.encoder.impl.SurfaceEncoder;
import de.datexis.encoder.impl.TrigramEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.ner.MentionAnnotator;
import de.datexis.ner.reader.CoNLLDatasetReader;
import java.io.IOException;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("texoo-train-ner", "TeXoo: train MentionAnnotator with CoNLL annotations", params.setUpCliOptions(), "", true);
    }
   
  }
  
  protected static class ExecParams implements CommandLineParser.Options {

    protected String trainingPath;
    protected String validationPath;
    protected String testPath;
    protected String outputPath;
    protected String language = "en";
    protected boolean trainingUI = false;

    @Override
    public void setParams(CommandLine parse) {
      trainingPath = parse.getOptionValue("i");
      validationPath = parse.getOptionValue("v");
      testPath = parse.getOptionValue("t");
      outputPath = parse.getOptionValue("o");
      trainingUI = parse.hasOption("u");
    }

    protected void TrainMentionAnnotatorCoNLL() {}
    
    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "input", true, "path to input training data (CoNLL format)");
      op.addOption("v", "validation", true, "path to validation data (CoNLL format)");
      op.addOption("t", "test", true, "path to test data (CoNLL format)");
      op.addRequiredOption("o", "output", true, "path to create and store the model");
      op.addOption("l", "language", true, "language to use for sentence splitting and stopwords (EN or DE)");
      op.addOption("u", "ui", false, "enable training UI (http://127.0.0.1:9000)");
      return op;
    }

  }
  
  protected void runTraining(ExecParams params) throws IOException {
    
    // Configure parameters
    Resource trainingPath = Resource.fromDirectory(params.trainingPath);
    //Resource validationPath = Resource.fromDirectory(params.validationPath);
    //Resource testPath = Resource.fromDirectory(params.testPath);
    Resource output = Resource.fromDirectory(params.outputPath);
    Language lang = WordHelpers.getLanguage(params.language);
    
    // Read datasets
    Dataset train = CoNLLDatasetReader.readDataset(trainingPath, trainingPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);
    //Dataset validation = CoNLLDatasetReader.readDataset(validationPath, validationPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);
    //Dataset test = CoNLLDatasetReader.readDataset(testPath, testPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);

    // Configure model
    MentionAnnotator ner = new MentionAnnotator.Builder()
        .withEncoders("tri", new PositionEncoder(), new SurfaceEncoder(), new TrigramEncoder())
        .enableTrainingUI(params.trainingUI)
        .pretrain(train)
        .build();

    // Train model
    ner.trainModel(train, Annotation.Source.GOLD, lang, -1, false, true);
    
    // Save model
    ner.writeModel(output);
    
  }
  
}
