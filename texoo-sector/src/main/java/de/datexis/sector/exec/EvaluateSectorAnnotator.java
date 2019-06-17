package de.datexis.sector.exec;

import de.datexis.annotator.AnnotatorFactory;
import de.datexis.common.CommandLineParser;
import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import de.datexis.sector.SectorAnnotator;
import de.datexis.sector.reader.WikiSectionReader;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Run experiments on a pre-trained SECTOR model
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EvaluateSectorAnnotator {
  
  protected final static Logger log = LoggerFactory.getLogger(EvaluateSectorAnnotator.class);
  
  public static void main(String[] args) throws IOException {
    
    final EvaluateSectorAnnotator.ExecParams params = new EvaluateSectorAnnotator.ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
      new EvaluateSectorAnnotator().evaluate(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("texoo-evaluate-sector", "TeXoo: evaluate SectorAnnotator on WikiSection dataset", params.setUpCliOptions(), "", true);
      System.exit(1);
    } catch(Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
    
  }
  
  protected static class ExecParams implements CommandLineParser.Options {
    
    protected String modelPath = null;
    protected String testFile = null;
    protected String embeddingsPath = null;
    protected boolean testSegmentation = false;
    
    @Override
    public void setParams(CommandLine parse) {
      modelPath = parse.getOptionValue("m");
      testFile = parse.getOptionValue("t");
      embeddingsPath = parse.getOptionValue("e");
      testSegmentation = parse.hasOption("s");
    }
    
    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("m", "model", true, "path to the pre-trained model");
      op.addRequiredOption("t", "test", true, "file name of WikiSection test dataset (will test after training if given)");
      op.addOption("e", "embedding", true, "search path to word embedding models (if not provided by the model itself)");
      op.addOption("s", "segment", false, "evaluate full segmentation model instead of faster sentence classification");
      return op;
    }
    
  }
  
  public void evaluate(EvaluateSectorAnnotator.ExecParams params) throws IOException {
    
    // Configure parameters
    Resource testFile = Resource.fromDirectory(params.testFile);
    Resource modelPath = Resource.fromDirectory(params.modelPath);
    Resource embeddingsPath = params.embeddingsPath != null ? Resource.fromDirectory(params.embeddingsPath) : null;
    
    // Load model
    SectorAnnotator sector = (SectorAnnotator) (params.embeddingsPath != null ?
      AnnotatorFactory.loadAnnotator(modelPath, embeddingsPath) :
      AnnotatorFactory.loadAnnotator(modelPath));
  
    // Read dataset
    Dataset test = WikiSectionReader.readDatasetFromJSON(testFile);
    
    // Annotate documents
    //sector.getTagger().setBatchSize(8); // if you need to save RAM on CUDA device
    // will attach SectorEncoder vectors to Sentences and create SectionAnnotations
    // Evaluate annotated documents for segmentation and segment classification
    if(params.testSegmentation) {
      log.info("Testing full BEMD segmentation model (might take longer)");
      sector.annotate(test.getDocuments(), SectorAnnotator.SegmentationMethod.BEMD);
      sector.evaluateModel(test, false, true, true);
    } else {
      log.info("Testing sentence classification (fast, but no segmentation)");
      sector.annotate(test.getDocuments(), SectorAnnotator.SegmentationMethod.NONE);
      sector.evaluateModel(test, true, false, false);
    }
    
  }
  
}