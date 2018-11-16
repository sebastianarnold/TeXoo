package de.datexis.ner.exec;

import de.datexis.common.CommandLineParser;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.impl.PositionEncoder;
import de.datexis.encoder.impl.SurfaceEncoder;
import de.datexis.encoder.impl.TrigramEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.ner.MatchingAnnotator;
import de.datexis.ner.MentionAnnotator;
import de.datexis.reader.RawTextDatasetReader;
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
public class TrainMentionAnnotatorSeedList {

  protected final static Logger log = LoggerFactory.getLogger(TrainMentionAnnotatorSeedList.class);

  public static void main(String[] args) throws IOException {
    
    final ExecParams params = new ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
      new TrainMentionAnnotatorSeedList().runTraining(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("texoo-train-ner-seed", "TeXoo: train MentionAnnotator with seed list", params.setUpCliOptions(), "", true);
      System.exit(1);
    }
   
  }
  
  protected static class ExecParams implements CommandLineParser.Options {

    protected String inputFiles;
    protected String seedList;
    protected String language;
    protected String outputPath = null;
    protected boolean trainingUI = false;

    @Override
    public void setParams(CommandLine parse) {
      inputFiles = parse.getOptionValue("i");
      seedList = parse.getOptionValue("s");
      outputPath = parse.getOptionValue("o");
      trainingUI = parse.hasOption("u");
      language = parse.getOptionValue("l", "en");
    }

    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "input", true, "path or file name for raw input text");
      op.addRequiredOption("s", "seed", true, "path to seed list text file");
      op.addRequiredOption("o", "output", true, "path to create and store the model");
      op.addOption("l", "language", true, "language to use for sentence splitting and stopwords (EN or DE)");
      op.addOption("u", "ui", false, "enable training UI (http://127.0.0.1:9000)");
      return op;
    }
    
  }
    
  protected void runTraining(ExecParams params) throws IOException {

    // Configure parameters
    Resource inputPath = Resource.fromDirectory(params.inputFiles);
    //Resource validationPath = Resource.fromDirectory(params.validationPath);
    //Resource testPath = Resource.fromDirectory(params.testPath);
    Resource outputPath = Resource.fromDirectory(params.outputPath);
    Resource seedPath = Resource.fromDirectory(params.seedList);
    WordHelpers.Language lang = WordHelpers.getLanguage(params.language);

    // Read datasets
    Dataset train =new RawTextDatasetReader().read(inputPath);
    //Dataset validation = CoNLLDatasetReader.readDataset(validationPath, validationPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);
    //Dataset test = CoNLLDatasetReader.readDataset(testPath, testPath.getFileName(), CoNLLDatasetReader.Charset.UTF_8);

    // Configure matcher
    MatchingAnnotator match = new MatchingAnnotator(MatchingAnnotator.MatchingStrategy.LOWERCASE);
    match.loadTermsToMatch(seedPath);

    // Configure model
    MentionAnnotator ner = new MentionAnnotator.Builder()
      .withEncoders("tri", new PositionEncoder(), new SurfaceEncoder(), new TrigramEncoder())
      .enableTrainingUI(params.trainingUI)
      .withTrainingParams(0.0001, 16, 1)
      .withModelParams(512, 256)
      .withWorkspaceParams(1) // single worker
      .pretrain(train)
      .build();

    // Train model
    ner.trainModel(train, Annotation.Source.SILVER, lang, 5000, false, true);

    // Save model
    System.out.println("saving model to path: " + outputPath);
    ner.writeModel(outputPath);

  }

}