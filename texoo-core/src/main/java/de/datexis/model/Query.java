package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

import java.util.*;
import java.util.stream.Collectors;

@JsonPropertyOrder({ "class", "id", "uid", "refUid", "title", "language", "type", "begin", "length", "text", "annotations", "documentRelevance" })
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "class", defaultImpl=Query.class)
@JsonIgnoreProperties(ignoreUnknown = true)
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
  @JsonIgnore
  public IRRelevance getIRRelevance(Document document){
    return relevantDocuments.getOrDefault(document, new IRScore(0.0));
  }

  /**
   * Checks if an document is relevant
   * @param document The Document
   * @return If the document is relevant
   */
  @JsonIgnore
  public boolean isRelevant(Document document){
    return relevantDocuments.containsKey(document);
  }

  public void addRelevantDocument(Document document, IRRelevance relevance){
    relevantDocuments.put(document, relevance);
  }

  @JsonIgnore
  public Map<Document, IRRelevance> getRelevantDocuments() {
    return relevantDocuments;
  }

  public List<IRDocumentRelevance> getDocumentRelevance(){
    return relevantDocuments.entrySet().stream()
      .map(e -> new IRDocumentRelevance(e.getKey(), e.getValue()))
      .collect(Collectors.toList());
  }

  public void setDocumentRelevance(List<IRDocumentRelevance> documentRelevance){
    documentRelevance.stream()
      .forEach(dr -> relevantDocuments.put(dr.document, dr.relevance));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof Query)) {
      return false;
    }
    Query query = (Query) o;
    return super.equals(query) &&
      Objects.equals(relevantDocuments, query.relevantDocuments);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), relevantDocuments);
  }

  @Override
  public String toString() {
    return "Querry [sentences=" + this.sentences + ", relevantDocuments=" + this.relevantDocuments + "]";
  }

  @JsonPropertyOrder({ "class", "document", "relevance" })
  @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "class", defaultImpl=IRDocumentRelevance.class)
  @JsonIgnoreProperties(ignoreUnknown = true)
  public static class IRDocumentRelevance{
    private Document document;
    private IRRelevance relevance;

    public IRDocumentRelevance(Document document, IRRelevance relevance) {
      this.document = document;
      this.relevance = relevance;
    }

    public Document getDocument() {
      return document;
    }

    public void setDocument(Document document) {
      this.document = document;
    }

    public IRRelevance getRelevance() {
      return relevance;
    }

    public void setRelevance(IRRelevance relevance) {
      this.relevance = relevance;
    }
  }
}
