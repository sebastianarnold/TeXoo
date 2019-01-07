package de.datexis.model;

import java.util.ArrayList;
import java.util.List;

public class IRDataset {
  /**
   * All Documents that will be queried
   */
  private List<Document> documents;

  /**
   * All queries that will be aplied
   */
  private List<Query> queries;

  /**
   * The name of the Dataset
   */
  private String name;

  public IRDataset(String name) {
    this.name = name;

    documents = new ArrayList<>();
    queries = new ArrayList<>();
  }

  public void addDocument(Document document){
    documents.add(document);
  }

  public void addQuery(Query query){
    queries.add(query);
  }

  public List<Document> getDocuments() {
    return documents;
  }

  public List<Query> getQueries() {
    return queries;
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }
}
