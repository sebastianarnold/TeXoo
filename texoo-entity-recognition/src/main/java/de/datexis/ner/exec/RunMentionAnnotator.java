package de.datexis.ner.exec;

import de.datexis.common.CommandLineParser;
import de.datexis.common.Configuration;
import java.io.IOException;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.deeplearning4j.ui.api.UIServer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main Controller for training of MentionAnnotator / NER models.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class RunMentionAnnotator {

  protected final static Logger log = LoggerFactory.getLogger(RunMentionAnnotator.class);

  public static void main(String[] args) throws IOException {
    
    final ExecParams params = new ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("texoo-annotate-ner", "TeXoo: run pre-trained MentionAnnotator model", params.setUpCliOptions(), "", true);
    }
   
  }
  
  protected static class ExecParams implements CommandLineParser.Options {

    protected String inputFiles;
    protected String seedList;
    protected String outputPath = null; //"/home/sarnold/Library/Models/TeXoo/sector_2.0/SectorAnnotator_de_disease_SECTOR+bloom_20180515/head/";
    protected boolean trainingUI = true;

    @Override
    public void setParams(CommandLine parse) {
      inputFiles = parse.getOptionValue("i");
      seedList = parse.getOptionValue("s");
      outputPath = parse.getOptionValue("o");
      trainingUI = parse.hasOption("u");
    }

    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "input", true, "path and file name pattern for raw input text");
      op.addRequiredOption("s", "seed", true, "path to seed list text file");
      op.addRequiredOption("o", "output", true, "path to create and store the model");
      op.addOption("u", "ui", false, "enable training UI");
      return op;
    }

  }
  
}
