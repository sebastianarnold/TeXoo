package de.datexis.cdv.eval;

import de.datexis.annotator.AnnotatorFactory;
import de.datexis.cdv.CDVAnnotator;
import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.index.AspectIndexBuilder;
import de.datexis.cdv.index.EntityIndex;
import de.datexis.cdv.index.PassageIndex;
import de.datexis.cdv.reader.MatchZooReader;
import de.datexis.cdv.retrieval.QueryRunner;
import de.datexis.common.CommandLineParser;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import de.datexis.retrieval.eval.RetrievalEvaluation;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * WWW2020: Use pre-trained CDV model for retrieval
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EvaluateCDVRetrieval {

  protected final static Logger log = LoggerFactory.getLogger(EvaluateCDVRetrieval.class);
  
  public static void main(String[] args) throws IOException {
  
    final EvaluateCDVRetrieval.ExecParams params = new EvaluateCDVRetrieval.ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
  
    try {
      parser.parse(args);
      if(params.multiTaskModelDir != null) new EvaluateCDVRetrieval().evalMultiTaskRetrieval(params);
      else new EvaluateCDVRetrieval().evalSingleTaskRetrieval(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("evaluate-cdv", "TeXoo: evaluate CDV answer retrieval", params.setUpCliOptions(), "", true);
      System.exit(1);
    } catch(Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
    
  }
  
  protected static class ExecParams implements CommandLineParser.Options {
    
    protected String multiTaskModelDir = null;
    protected String entityModelDir = null;
    protected String aspectModelDir = null;
    protected String encoderDir = null;
    protected String datasetDir = null;
    
    @Override
    public void setParams(CommandLine parse) {
      multiTaskModelDir = parse.getOptionValue("m");
      entityModelDir = parse.getOptionValue("e");
      aspectModelDir = parse.getOptionValue("a");
      encoderDir = parse.getOptionValue("p");
      datasetDir = parse.getOptionValue("d");
    }
    
    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("m", "model", true, "path to the pre-trained CDV multi-task model");
      op.addOption("e", "entity", true, "optional path to a CDV single-task entity model");
      op.addOption("a", "aspect", true, "optional path to a CDV single-task aspect model");
      op.addOption("p", "path", true, "search path to sentence embedding models (if not provided by the model itself)");
      op.addRequiredOption("d", "dataset", true, "path to the evaluation dataset (json)");
      return op;
    }
    
  }
  
  public EvaluateCDVRetrieval() {}
  
  public void evalSingleTaskRetrieval(EvaluateCDVRetrieval.ExecParams params) throws IOException {
    
    Resource datasetPath = Resource.fromDirectory(params.datasetDir);
    Resource entityModelPath = Resource.fromDirectory(params.entityModelDir);
    Resource aspectModelPath = Resource.fromDirectory(params.aspectModelDir);
    Resource embeddingPath = Resource.fromDirectory(params.encoderDir);
    
    // --- load data ---------------------------------------------------------------------------------------------------
    Dataset corpus = ObjectSerializer.readFromJSON(datasetPath, Dataset.class);
    
    // --- load model --------------------------------------------------------------------------------------------------
    CDVAnnotator entityAnnotator = (CDVAnnotator) AnnotatorFactory.loadAnnotator(entityModelPath, embeddingPath);
    CDVAnnotator aspectAnnotator = (CDVAnnotator) AnnotatorFactory.loadAnnotator(aspectModelPath, embeddingPath);
    EntityIndex entityIndex = (EntityIndex) entityAnnotator.getEntityEncoder();
    AspectIndex aspectIndex = AspectIndexBuilder.buildAspectIndex(aspectAnnotator.getAspectEncoder(), corpus.getName());
    
    // --- annotate ----------------------------------------------------------------------------------------------------
    entityAnnotator.annotateDocuments(corpus.getDocuments());
    aspectAnnotator.annotateDocuments(corpus.getDocuments());
    
    // --- query ----------------------------------------------------------------------------------------------------
    QueryRunner runner = new QueryRunner(corpus, entityIndex, aspectIndex, QueryRunner.Strategy.PASSAGE_RANK);
    MatchZooReader.addCandidateSamples(corpus, PassageIndex.NUM_CANDIDATES); // adds 64 candidates to be comparable with MatchZoo models
    runner.retrieveAllQueries(QueryRunner.Candidates.GIVEN);
  
    // --- evaluate ----------------------------------------------------------------------------------------------------
    RetrievalEvaluation eval = new RetrievalEvaluation(corpus.getName());
    eval.evaluateQueries(corpus);
    eval.printEvaluationStats();
    
  }
  
  public void evalMultiTaskRetrieval(EvaluateCDVRetrieval.ExecParams params) throws IOException {
    
    Resource datasetPath = Resource.fromDirectory(params.datasetDir);
    Resource cdvModelPath = Resource.fromDirectory(params.multiTaskModelDir);
    Resource embeddingPath = Resource.fromDirectory(params.encoderDir != null ? params.encoderDir : params.multiTaskModelDir);
    
    // --- load data ---------------------------------------------------------------------------------------------------
    Dataset corpus = ObjectSerializer.readFromJSON(datasetPath, Dataset.class);
    
    // --- load model --------------------------------------------------------------------------------------------------
    CDVAnnotator cdv = (CDVAnnotator) AnnotatorFactory.loadAnnotator(cdvModelPath, embeddingPath);
    EntityIndex entityIndex = (EntityIndex) cdv.getEntityEncoder();
    AspectIndex aspectIndex = AspectIndexBuilder.buildAspectIndex(cdv.getAspectEncoder(), corpus.getName());
  
    // --- annotate ----------------------------------------------------------------------------------------------------
    cdv.getTagger().setMaxWordsPerSentence(-1); // don't limit sentence length during inference
    cdv.getTagger().setMaxTimeSeriesLength(-1); // don't limit document length during inference
    cdv.getTagger().setBatchSize(16);
    cdv.annotateDocuments(corpus.getDocuments());
    
    // --- query ----------------------------------------------------------------------------------------------------
    QueryRunner runner = new QueryRunner(corpus, entityIndex, aspectIndex, QueryRunner.Strategy.PASSAGE_RANK);
    MatchZooReader.addCandidateSamples(corpus, PassageIndex.NUM_CANDIDATES); // adds 64 candidates to be comparable with MatchZoo models
    runner.retrieveAllQueries(QueryRunner.Candidates.GIVEN);
    
    // --- evaluate ----------------------------------------------------------------------------------------------------
    RetrievalEvaluation eval = new RetrievalEvaluation(corpus.getName());
    eval.evaluateQueries(corpus);
    eval.printEvaluationStats();
  
  }
  
}