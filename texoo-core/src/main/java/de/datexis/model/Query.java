package de.datexis.model;

import java.util.HashMap;
import java.util.Map;

public class Query extends Document {
  /**
   * All relevant Documents with their relevanc score or rank
   */
  private Map<Document, IRRelevance> relevantDocuments;

  public Query() {
    relevantDocuments = new HashMap<>();
  }

  /**
   * Gets the relevance of an Document
   * @param document The document
   * @return The relevance of the document
   */
  public IRRelevance getIRRelevance(Document document){
    return relevantDocuments.getOrDefault(document, new IRScore(0.0));
  }

  /**
   * Checks if an document is relevant
   * @param document The Document
   * @return If the document is relevant
   */
  public boolean isRelevant(Document document){
    return relevantDocuments.containsKey(document);
  }

  public void addRelevantDocument(Document document, IRRelevance relevance){
    relevantDocuments.put(document, relevance);
  }

  public Map<Document, IRRelevance> getRelevantDocuments() {
    return relevantDocuments;
  }
}
