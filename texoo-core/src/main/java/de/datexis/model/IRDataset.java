package de.datexis.model;

import com.google.common.base.Objects;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

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

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof IRDataset)) {
      return false;
    }
    IRDataset dataset = (IRDataset) o;
    return Objects.equal(documents, dataset.documents) &&
      Objects.equal(queries, dataset.queries) &&
      Objects.equal(name, dataset.name);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(documents, queries, name);
  }
}
