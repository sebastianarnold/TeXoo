package de.datexis.tagger;

import de.datexis.model.Dataset;
import de.datexis.model.Document;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.*;

/**
 * A MultiDatasetIterator that returns one Document per Example, with Sentences as time steps.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class DocumentSentenceIterator extends AbstractMultiDataSetIterator {
  
  protected List<Document> documents;
  protected Iterator<Document> docIt;
  
  public DocumentSentenceIterator(Stage stage, Dataset dataset, int batchSize, boolean randomize) {
    this(stage, dataset.getDocuments(), batchSize, randomize);
  }
  
  public DocumentSentenceIterator(Stage stage, Collection<Document> docs, int batchSize, boolean randomize) {
    this(stage, docs, -1, -1, batchSize, randomize);
  }
  
  public DocumentSentenceIterator(Stage stage, Collection<Document> docs, int numExamples, int maxTimeSeriesLength, int batchSize, boolean randomize) {
    super(stage, numExamples, maxTimeSeriesLength, batchSize, randomize);
    this.documents = new ArrayList<>(docs);
    this.numExamples = numExamples > 0 && numExamples <= documents.size() ? numExamples : documents.size();
  }
  
  @Override
  public void reset() {
    if(randomize) Collections.shuffle(documents, new Random(System.nanoTime()));
    super.reset();
    docIt = documents.iterator();
  }
  
  protected boolean hasNextDocument() {
    return docIt != null && docIt.hasNext();
  }
  
  @Override
  public boolean hasNext() {
    return hasNextDocument() && !reachedEnd();
  }
  
  protected Document nextDocument() {
    cursor++;
    return docIt.next();
  }

  /**
   * Returns the next batch of documents.
   * @param num - batch size
   * @return List of Documents and the size of the longest document (in Sentences)
   */
  protected DocumentBatch nextBatch(int num) {
    Document example;
    ArrayList<Document> examples = new ArrayList<>(num);
    int maxWords = 1, maxSents = 1; // guarantee to to not return a zero-size dataset
    for(int batchNum=0; batchNum<num; batchNum++) {
      if(hasNext()) example = nextDocument();
      else example = new Document();
      examples.add(example);
      if(maxTimeSeriesLength > 0) maxSents = Math.min(Math.max(maxSents, example.countSentences()), maxTimeSeriesLength);
      else maxSents = Math.max(maxSents, example.countSentences());
      OptionalInt longestSentence = example.streamSentences().mapToInt(s -> s.countTokens()).max();
      maxWords = Math.max(maxWords, longestSentence.orElse(1));
    }
    return new DocumentBatch(num, examples, maxSents, maxWords, null);
  }
  
  public class DocumentBatch {
    public List<Document> docs;
    public MultiDataSet dataset;
    public int size;
    public int maxDocLength;
    public int maxSentenceLength;
    public DocumentBatch(int batchSize, List<Document> docs, int maxDocLength, int maxSentenceLength, MultiDataSet dataset) {
      this.size = batchSize;
      this.docs = docs;
      this.dataset = dataset;
      this.maxDocLength = maxDocLength;
      this.maxSentenceLength = maxSentenceLength;
    }
  }
  
  @Override
  public MultiDataSet next(int num) {
    DocumentBatch batch = nextDocumentBatch(num);
    return batch.dataset;
  }
  
  public DocumentBatch nextDocumentBatch() {
    return nextDocumentBatch(batchSize);
  }
  
  public DocumentBatch nextDocumentBatch(int num) {
    DocumentBatch batch = nextBatch(num);
    batch.dataset = generateDataSet(batch);
    reportProgress(batch.maxDocLength);
    return batch;
  }
  
  public abstract MultiDataSet generateDataSet(DocumentBatch batch);
  
}
