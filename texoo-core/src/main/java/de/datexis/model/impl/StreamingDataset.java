package de.datexis.model.impl;

import de.datexis.model.Dataset;
import de.datexis.model.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A Dataset that streams Documents from a provided Reader.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class StreamingDataset extends Dataset {
  
  protected final static Logger log = LoggerFactory.getLogger(StreamingDataset.class);
  
  private Stream<Document> documents;
  
  public StreamingDataset(Stream<Document> documents) {
    this.documents = documents;
  }
  
  /**
   * @return all Documents in this Dataset in no particular order
   */
  @Override
  public Collection<Document> getDocuments() {
    return documents.collect(Collectors.toList());
  }
  
  /**
   * @return a Stream of all Documents in this Dataset
   */
  @Override
  public Stream<Document> streamDocuments() {
    return documents;
  }
  
  /**
   * Add a document to the end of this Dataset
   */
  @Override
  public void addDocument(Document doc) {
    throw new UnsupportedOperationException("Cannot add to StreamingDataset");
  }
  
  @Override
  public void addDocumentFront(Document d) {
    throw new UnsupportedOperationException("Cannot add to StreamingDataset");
  }
  
  /**
   * @return the number of Documents in this Dataset
   */
  @Override
  public int countDocuments() {
    throw new UnsupportedOperationException("Cannot count Documents in StreamingDataset");
  }
  
}
