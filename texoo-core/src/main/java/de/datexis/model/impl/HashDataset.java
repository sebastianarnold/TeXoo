package de.datexis.model.impl;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * A Dataset that is optimized to access Documents using their IDs.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class HashDataset extends Dataset {
  
  protected final static Logger log = LoggerFactory.getLogger(HashDataset.class);
  
  /** A map of all Documents in this dataset, referenced by their ID */
  private Multimap<String, Document> documentIndex;
  
  public HashDataset(String name, Collection<Document> docs) {
    this.documentIndex = ArrayListMultimap.create();
    docs.stream().forEach(doc -> addDocument(doc));
    setName(name);
  }
  
  /**
   * Add a document to the end of this Dataset
   */
  public void addDocument(Document doc) {
    if(getLanguage() == null) setLanguage(doc.getLanguage());
    documentIndex.put(doc.getId(),doc);
  }
  
  public void addDocumentFront(Document d) {
    throw new UnsupportedOperationException("HashDataset has no Document order.");
  }
  
  /**
   * Find a Document with given ID in the Dataset.
   * If multiple Documents exist with the same ID, only one is returned.
   * @return the Document with given ID
   */
  public Optional<Document> getDocument(String id) {
    return documentIndex.get(id).stream().findFirst();
  }
  
  /**
   * Update all Indexes in case Document IDs have changed
   */
  public void updateIndexes() {
    // TODO: implement
  }
  
  /**
   * @return the number of Documents in this Dataset
   */
  public int countDocuments() {
    return documentIndex.size();
  }
  
  /**
   * @return all Documents in this Dataset in no particular order
   */
  public Collection<Document> getDocuments() {
    return documentIndex.values();
  }
  
  /**
   * @return a Stream of all Documents in this Dataset
   */
  public Stream<Document> streamDocuments() {
    return documentIndex.values().stream();
  }
  
  public void randomizeDocuments() {
    throw new UnsupportedOperationException("HashDataset cannot be randomized.");
  }
  
  public void randomizeDocuments(long seed) {
    throw new UnsupportedOperationException("HashDataset cannot be randomized.");
  }
  
}
