package de.datexis.sector.tagger;

import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.tagger.AbstractMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.*;

/**
 * A MultiDatasetIterator that returns one Document per Example, with Sentences as time steps.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class DocumentSentenceIterator extends AbstractMultiDataSetIterator {

  public DocumentSentenceIterator(Stage stage, Dataset dataset, int batchSize, boolean randomize) {
    this(stage, dataset.getDocuments(), batchSize, randomize);
  }
  
  public DocumentSentenceIterator(Stage stage, Collection<Document> docs, int batchSize, boolean randomize) {
    this(stage, docs, -1, -1, batchSize, randomize);
  }
  
  public DocumentSentenceIterator(Stage stage, Collection<Document> docs, int numExamples, int maxTimeSeriesLength, int batchSize, boolean randomize) {
    super(stage, docs, numExamples, maxTimeSeriesLength, batchSize, randomize);
  }
  
  @Override
  public void reset() {
    cursor = 0;
    if(randomize) Collections.shuffle(documents, new Random(System.nanoTime()));
    docIt = documents.iterator();
    startTime = System.currentTimeMillis();
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
    int exampleSize = 1; // guarantee to to not return a zero-size dataset
    for(int batchNum=0; batchNum<num; batchNum++) {
      if(hasNext()) example = nextDocument();
      else example = new Document();
      examples.add(example);
      if(maxTimeSeriesLength > 0) exampleSize = Math.min(Math.max(exampleSize, example.countSentences()), maxTimeSeriesLength);
      else exampleSize = Math.max(exampleSize, example.countSentences());
    }
    return new DocumentBatch(num, examples, exampleSize, null);
  }
  
  public class DocumentBatch {
    public List<Document> docs;
    public MultiDataSet dataset;
    public int size;
    public int maxDocLength;
    public DocumentBatch(int batchSize, List<Document> docs, int maxDocLength, MultiDataSet dataset) {
      this.size = batchSize;
      this.docs = docs;
      this.dataset = dataset;
      this.maxDocLength = maxDocLength;
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
