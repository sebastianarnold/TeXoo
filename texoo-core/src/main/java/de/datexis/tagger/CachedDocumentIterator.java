package de.datexis.tagger;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.stream.StreamSupport;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.dataset.DataSet;

/**
 * A DataSetIterator which holds all neccessary functions to iterate thorugh Documents or Sentences.
 * Will encode and cache all required encodings in the documents.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class CachedDocumentIterator extends AbstractIterator {

  protected Iterator<Document> docIt = null;
  protected Iterator<Sentence> sentIt = null;
  
  public CachedDocumentIterator(Collection<Document> docs, String name, int numExamples, int batchSize, boolean randomize) {
    super(docs, name, numExamples, batchSize, randomize);
    this.totalExamples = (int) StreamSupport.stream(docs.spliterator(), false).count();
    this.numExamples = numExamples < 0 ? this.totalExamples : numExamples;  
  }
  
  @Override
  public boolean asyncSupported() {
    return false; // we call clearCache!
  }
  
  @Override
  public final void reset() {
    cursor = 0;
    if(randomize) documents = randomizeDocuments(documents);
    docIt = documents.iterator();
    startTime = System.currentTimeMillis();
  }
  
  private boolean reachedEnd() {
    return cursor >= numExamples;
  }

  @Override
  public boolean hasNext() {
    return hasNextDocument() && !reachedEnd();
  }
  
  protected boolean hasNextDocument() {
    return docIt != null && docIt.hasNext();
  }
  
  protected Document nextDocument() {
    cursor++;
    currDocument = docIt.next();
    //encodeDocument(currDocument);
    return currDocument;
  }
  
  /**
   * Process all Encoders in given order and attach Embeddings to Tokens.
   */
  protected abstract void encodeDocument(Document d);
  
  protected void encodeDocuments(Iterable<Document> docs) {
    for(Document d : docs) {
      encodeDocument(d);
    }
  }

  protected abstract void clearCachedDocument(Document d);
  
  protected boolean clearCache() {
    boolean cleared = false;
    for(Document d : docsInUse) {
      if(d == currDocument) continue;
      clearCachedDocument(d);
      cleared = true;
    }
    if(cleared) {
      docsInUse.clear();
      docsInUse.add(currDocument);
      //System.gc();
    }
    return cleared;
  }
  
  @Override
  public DataSet next(int num) {
    Pair<ArrayList<Document>,Integer> batch = nextBatchOfDocuments(num);
    reportProgress();
    DataSet result = generateDataSet(batch.getKey(), num, batch.getValue());
    log.trace("Iterate: example size " + num + " Documents x " + batch.getValue() + " Sentences");
    return result;
  }
  
  public Pair<DataSet,ArrayList<Document>> nextDataSet() {
    return nextDataSet(batchSize);
  }
   
  public Pair<DataSet,ArrayList<Document>> nextDataSet(int num) {
    Pair<ArrayList<Document>,Integer> batch = nextBatchOfDocuments(num);
    reportProgress();
    DataSet result = generateDataSet(batch.getKey(), num, batch.getValue());
    log.trace("Iterate: example size " + num + " Documents x " + batch.getValue() + " Sentences");
    return new ImmutablePair<>(result, batch.getKey());
  }
  
  /**
   * Returns the next batch of documents.
   * @param num - batch size
   * @return List of Documents and the size of the longest document (in Tokens)
   */
  public Pair<ArrayList<Document>,Integer> nextBatchOfDocuments(int num) {
    Document example;
    ArrayList<Document> examples = new ArrayList<>(num);
    int exampleSize = 1; // guarantee to to not return a zero-size dataset
    for(int batchNum=0; batchNum<num; batchNum++) {
      if(hasNext()) example = nextDocument();
      else example = new Document();
      examples.add(example);
      exampleSize = Math.max(exampleSize, example.countSentences()); // FIXME: we need to generate Marker Tokens as well
    }
    encodeDocuments(examples);
    return new ImmutablePair<>(examples, exampleSize);
  }
  
  public abstract DataSet generateDataSet(ArrayList<Document> examples, int num, int exampleSize);
  
  
}
