package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Holds a collection of Documents in memory(!)
 * @author sarnold
 */
public class Dataset {
  
  private static final Logger log = LoggerFactory.getLogger(Dataset.class);
  
  /** The name of this dataset */
  private String name;
  
  /** The language of this dataset */
  private String language = null;
  
  /** The unique ID of this dataset (e.g. database primary key) */
  private Long uid = null;
  
  /** A list of all Documents in this dataset */
  private List<Document> documents;
  
  /** A list of Queries and their Results on this dataset */
  List<Query> queries = new ArrayList<>();
  
  /** Random seed */
  protected static Random random = new Random();
  
  public Dataset() {
    this("");
  }
  
  public Dataset(String name) {
    this(name, new ArrayList<>());
  }
  
  public Dataset(String name, List<Document> docs) {
    this.documents = docs;
    this.name = name;
  }
  
  public String getName() {
    return name;
  }
  
  public void setName(String name) {
    this.name = name;
  }
  
  @JsonInclude(JsonInclude.Include.NON_NULL)
  public String getLanguage() {
    return language;
  }
  
  public void setUid(Long uid) {
    this.uid = uid;
  }
  
  @JsonInclude(JsonInclude.Include.NON_NULL)
  public Long getUid() {
    return this.uid;
  }

  public void setLanguage(String language) {
    this.language = language;
  }
  
  /**
   * @return a Dataset that references to a split of documents. Caution: this is not a deep copy.
   */
  public Dataset getSplit(int offset, int count) {
    if(offset < 0) offset = countDocuments() + offset;
    if(count < 0) count = countDocuments() + count;
    List<Document> docs = streamDocuments(offset, count).collect(Collectors.toList());
    return new Dataset(getName(), docs);
  }
  
  /**
   * @return all Documents in this Dataset in no particular order
   */
  public Collection<Document> getDocuments() {
    return documents;
  }
  
  /**
   * @return a Stream of all Documents in this Dataset
   */
  public Stream<Document> streamDocuments() {
    return documents.stream();
  }
  
  /**
   * @return a subset of Documents in this Dataset
   */
  public List<Document> getDocuments(int startIndex, int count) {
    return streamDocuments(startIndex, count)
      .collect(Collectors.toList());
  }
  
  /**
   * @return a Stream of Documents
   */
  public Stream<Document> streamDocuments(int startIndex, int count) {
    return streamDocuments()
      .skip(startIndex)
      .limit(count);
  }
  
  /**
   * @return the Document with given index
   */
  public Optional<Document> getDocument(int index) {
     return streamDocuments()
       .skip(index)
       .findFirst();
  }
  
  /**
   * Find a Document with given ID in the Dataset.
   * If multiple Documents exist with the same ID, only one is returned.
   * @return the Document with given ID
   */
  public Optional<Document> getDocument(String id) {
    return streamDocuments()
      .filter(doc -> doc.getId().equals(id))
      .findFirst();
  }
  
  /**
   * @return a random Document of this Dataset
   */
  @JsonIgnore
  public Optional<Document> getRandomDocument() {
    int index = random.nextInt(countDocuments());
    return getDocument(index);
  }

  public void randomizeDocuments() {
    Collections.shuffle(documents);
  }
  
  public void randomizeDocuments(long seed) {
    Collections.shuffle(documents, new Random(seed));
  }
  
  /**
   * @return stream over all Sentences in the Dataset. Caution: Boundaries are still given on Document level.
   */
  @JsonIgnore
  public Stream<Sentence> streamSentences() {
		return streamDocuments().flatMap(s -> s.streamSentences());
  }
  
  /**
   * @return stream over all Tokens in the Dataset. Caution: Boundaries are still given on Document level.
   */
  @JsonIgnore
  public Stream<Token> streamTokens() {
		return streamDocuments().flatMap(s -> s.streamTokens());
  }
  
  @JsonIgnore
  public <S extends Span> Stream<S> getStream(Class<S> spanClass) {
    if(spanClass == Sentence.class) return (Stream<S>) streamSentences();
    else if(spanClass == Token.class) return (Stream<S>) streamTokens();
    else return (Stream<S>) streamTokens();
  }
  
  /**
   * Add a document to the end of this Dataset
   */
  public void addDocument(Document doc) {
    if(language == null) setLanguage(doc.getLanguage());
    documents.add(doc);
  }
  
  public void addDocumentFront(Document d) {
    documents.add(0, d);
  }
  
  /**
   * @return the number of Documents in this Dataset
   */
  public int countDocuments() {
    return documents.size();
  }
  
  /**
   * @return the number of Sentences in all Documents in this Dataset
   */
  public long countSentences() {
    return streamDocuments().mapToLong(d -> d.countSentences()).sum();
  }
  
  /**
   * @return the number of Tokens in all Documents in this Dataset
   */
  public long countTokens() {
    return streamDocuments().mapToLong(d -> d.countTokens()).sum();
  }
  
  /**
   * @return the number of Annotations in all Documents in this Dataset
   */
  public long countAnnotations() {
    return streamDocuments().mapToLong(d -> d.countAnnotations()).sum();
  }
  
  /**
   * @return the number of Queries in this Dataset
   */
  public long countQueries() {
    return getQueries().size();
  }
  
  /**
   * @return the number of Annotations from a given source in all Documents in this Dataset
   */
  public long countAnnotations(Annotation.Source source) {
    return streamDocuments().mapToLong(d -> d.countAnnotations(source)).sum();
  }
  
  /**
   * @return the number of Annotations from a given source in all Documents in this Dataset
   */
  public <A extends Annotation> long countAnnotations(Annotation.Source source, Class<A> type) {
    return streamDocuments().mapToLong(d -> d.countAnnotations(source, type)).sum();
  }
  
  /**
   * @return a random Sentence from the Dataset
   */
  @JsonIgnore
  public Sentence getRandomSentence() {
    int index = random.nextInt(countDocuments());
    return getDocument(index).get().getRandomSentence();
  }
  
  public Collection<Query> getQueries() {
    return queries;
  }
  
  public void addQuery(Query q) {
    this.queries.add(q);
  }
  
  public Optional<Query> getQuery(String id) {
    return queries.stream()
      .filter(q -> q.getId().equals(id))
      .findFirst();
  }
  
  /**
   * @return a deep copy of this Dataset (not fully implemented yet!)
   */
  public Dataset clone() {
    ArrayList<Document> docs = new ArrayList<>(countDocuments());
    for(Document doc : getDocuments()) {
      docs.add(doc.clone());
    }
    return new Dataset(getName(), docs);
  }

  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof Dataset)) {
      return false;
    }
    
    Dataset dataset = (Dataset) o;
    return Objects.equals(getName(), dataset.getName()) &&
           Objects.equals(getLanguage(), dataset.getLanguage()) &&
           Objects.equals(getDocuments(), dataset.getDocuments());
  }

  @Override
  public int hashCode() {
    return Objects.hash(getName(), getLanguage(), getDocuments());
  }
}