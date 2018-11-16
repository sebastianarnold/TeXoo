package de.datexis.ner.exec;

import de.datexis.common.CommandLineParser;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.ner.GenericMentionAnnotator;
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
public class RunMentionAnnotator {

  protected final static Logger log = LoggerFactory.getLogger(RunMentionAnnotator.class);

  public static void main(String[] args) throws IOException {
    
    final ExecParams params = new ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
      new RunMentionAnnotator().runAnnotator(params);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("texoo-annotate-ner", "TeXoo: run pre-trained MentionAnnotator model", params.setUpCliOptions(), "", true);
    }
   
  }
  
  protected static class ExecParams implements CommandLineParser.Options {

    protected String inputFiles;
    protected String outputPath = null;
    protected String language = null;

    @Override
    public void setParams(CommandLine parse) {
      inputFiles = parse.getOptionValue("i");
      outputPath = parse.getOptionValue("o");
      //language = parse.getOptionValue("l");
    }

    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "input", true, "path or file name for raw input text");
      op.addOption("o", "output", true, "path to create and store the output JSON, otherwise dump to stdout");
      //op.addOption("l", "language", true, "language to use for annotation (EN or DE)");
      return op;
    }

  }
  
  protected void runAnnotator(ExecParams params) throws IOException {
    
    // Configure parameters
    Resource inputPath = Resource.fromDirectory(params.inputFiles);
    Resource outputPath = params.outputPath != null ? Resource.fromDirectory(params.outputPath) : null;
    //WordHelpers.Language lang = WordHelpers.getLanguage(params.language);
    
    // Read datasets
    // TODO: stream would be better so we can annotate file-by-file
    Dataset data = new RawTextDatasetReader().read(inputPath);

    // Load model
    MentionAnnotator ner = GenericMentionAnnotator.create();

    // Annotate
    // TODO: skip existing extractions unless -f is specified
    ner.annotate(data.getDocuments());
    
    // Save
    if(params.outputPath != null) {
      if(!outputPath.exists()) outputPath.toFile().mkdirs();
      for(Document doc : data.getDocuments()) {
        ObjectSerializer.writeJSON(doc, outputPath.resolve(doc.getId() + ".json"));
      }
    } else {
      System.out.print(ObjectSerializer.getJSON(data));
    }
    
  }
  
}
