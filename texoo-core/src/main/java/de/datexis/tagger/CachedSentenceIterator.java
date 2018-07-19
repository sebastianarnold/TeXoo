package de.datexis.tagger;

import de.datexis.encoder.Encoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.stream.StreamSupport;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.dataset.DataSet;

/**
 * A DataSetIterator which iterates through a Document, one Sentence per Example.
 * Will encode and cache all required encodings in the documents.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class CachedSentenceIterator extends AbstractIterator {

  protected Iterator<Document> docIt = null;
  protected Iterator<Sentence> sentIt = null;
  
  public CachedSentenceIterator(Collection<Document> docs, String name, int numExamples, int batchSize, boolean randomize) {
    super(docs, name, numExamples, batchSize, randomize);
    this.totalExamples = (int) StreamSupport.stream(docs.spliterator(), false).mapToInt(d -> d.countSentences()).sum();
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
    sentIt = null;
    startTime = System.currentTimeMillis();
  }
  
  private boolean reachedEnd() {
    return cursor >= numExamples;
  }
  
  @Override
  public boolean hasNext() {
    if(hasNextSentence()) {
      return !reachedEnd();
    } else if(hasNextDocument()) {
      sentIt = nextDocument().getSentences().iterator();
      return hasNext();
    } else {
      return false;
    }
  }
  
  public boolean hasNextDocument() {
    return docIt != null && docIt.hasNext();
  }
  
  public Document nextDocument() {
    currDocument = docIt.next();
    encodeDocument(currDocument);
    return currDocument;
  }
  
  public boolean hasNextSentence() {
    return sentIt != null && sentIt.hasNext();
  }
  
  public Sentence nextSentence() {
    cursor++;
    return sentIt.next();
  }
  
  /**
   * Process all Encoders in given order and attach Embeddings to Tokens.
   */
  protected void encodeDocument(Document d) {
    docsInUse.add(d);
    // TODO: we could check if the encoding already exists
    for(Encoder enc : encoders) {
      //log.info("Processing " + enc.getName() + " for " + d.countSentences() + " sentences...");
      enc.encodeEach(d, Token.class);
    }
  }
  
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
  
  protected void clearCachedDocument(Document d) {
    for(Token t : d.getTokens()) {
      t.clearVectors();
    }
    d.getAnnotations().forEach(a -> a.clearVectors());
    for(Sentence s : d.getSentences()) {
      s.clearVectors();
    }
  }
  
  @Override
  public DataSet next(int num) {
    Pair<ArrayList<Sentence>,Integer> batch = nextBatchOfSentences(num);
    reportProgress();
    DataSet result = generateDataSet(batch.getKey(), num, batch.getValue());
    log.trace("Iterate: example size " + num + " Sentences x " + batch.getValue() + " Tokens");
    return result;
  }
  
  public Pair<DataSet,ArrayList<Sentence>> nextDataSet() {
    return nextDataSet(batchSize);
  }
   
  public Pair<DataSet,ArrayList<Sentence>> nextDataSet(int num) {
    Pair<ArrayList<Sentence>,Integer> batch = nextBatchOfSentences(num);
    reportProgress();
    //num = batch.getKey().size(); // skip empty sentences at batch end
    DataSet result = generateDataSet(batch.getKey(), num, batch.getValue());
    log.trace("Iterate: example size " + num + " Sentences x " + batch.getValue() + " Tokens");
    return new ImmutablePair<>(result, batch.getKey());
  }
  
  public Pair<ArrayList<Sentence>,Integer> nextBatchOfSentences(int num) {
    Sentence example;
    ArrayList<Sentence> examples = new ArrayList<>(num);
    int exampleSize = 0;
    for(int batchNum=0; batchNum<num; batchNum++) {
      if(hasNext()) example = nextSentence();
      else example = new Sentence(); // else break;
      examples.add(example);
      exampleSize = Math.max(exampleSize, example.countTokens());
    }
    return new ImmutablePair<>(examples, exampleSize);
  }
  
  public abstract DataSet generateDataSet(ArrayList<Sentence> examples, int num, int exampleSize);
  
}
