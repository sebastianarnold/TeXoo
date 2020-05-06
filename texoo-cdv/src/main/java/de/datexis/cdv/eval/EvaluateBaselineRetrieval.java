package de.datexis.cdv.eval;

import de.datexis.cdv.index.PassageIndex;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import de.datexis.retrieval.eval.RetrievalEvaluation;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * WWW2020: Use Lucene Index for TF-IDF and BM25 baseline retrieval
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EvaluateBaselineRetrieval {

  protected final static Logger log = LoggerFactory.getLogger(EvaluateBaselineRetrieval.class);
  
  String datasetName = "MedQuAD";
  String datasetDir = "/home/sarnold/Library/Datasets/Heatmap/MatchZoo/" + datasetName + "-queries-test.json";

  public static void main(String[] args) throws IOException {
    try {
      new EvaluateBaselineRetrieval().evalRetrieval();
    } finally {
    }
  }
  
  public EvaluateBaselineRetrieval() {}
  
  public void evalRetrieval() throws IOException {
    
    Resource datasetPath = Resource.fromDirectory(datasetDir);
    
    // --- load data ---------------------------------------------------------------------------------------------------
    Dataset corpus = ObjectSerializer.readFromJSON(datasetPath, Dataset.class);
    
    // --- build index model -------------------------------------------------------------------------------------------
    PassageIndex index = new PassageIndex();
    index.createInMemoryIndex(corpus);
//    index.setSimilarity(new ClassicSimilarity());
    index.setSimilarity(new BM25Similarity());
    
    // --- query ----------------------------------------------------------------------------------------------------
    index.retrieveAllQueries(corpus, 64, false);
  
    // --- evaluate ----------------------------------------------------------------------------------------------------
  
    RetrievalEvaluation eval = new RetrievalEvaluation(corpus.getName());
    eval.evaluateQueries(corpus);
    eval.printEvaluationStats();
    
  }
  
}