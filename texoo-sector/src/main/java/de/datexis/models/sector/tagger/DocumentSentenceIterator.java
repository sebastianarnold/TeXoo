package de.datexis.models.sector.tagger;

import de.datexis.model.Dataset;
import de.datexis.model.Document;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class DocumentSentenceIterator implements MultiDataSetIterator {

  protected Logger log = LoggerFactory.getLogger(DocumentSentenceIterator.class);

  public static enum Stage { TRAIN, TEST, ENCODE };
  
  protected List<Document> documents;
  protected Iterator<Document> docIt;
  
  //protected List<Encoder> inputEncoders;
  //protected List<Encoder> outputEncoders;

  protected int numExamples;
  protected int batchSize;
  protected int cursor;
  protected long startTime;
  protected boolean randomize;
  
  protected Stage stage;
  
  public DocumentSentenceIterator(Stage stage, Dataset dataset, int batchSize, boolean randomize) {
    this(stage, dataset.getDocuments(), batchSize, randomize);
  }
  
  public DocumentSentenceIterator(Stage stage, Collection<Document> docs, int batchSize, boolean randomize) {
    this(stage, docs, -1, batchSize, randomize);
  }
  
  public DocumentSentenceIterator(Stage stage, Collection<Document> docs, int numExamples, int batchSize, boolean randomize) {
    this.documents = new ArrayList<>(docs);
    this.numExamples = numExamples > 0 && numExamples <= documents.size() ? numExamples : documents.size();
    this.batchSize = batchSize;
    this.randomize = randomize;
    this.stage = stage;
  }
  
  @Override
  public boolean asyncSupported() {
    return true;
  }
  
  @Override
  public boolean resetSupported() {
    return true;
  }
  
  @Override
  public final void reset() {
    cursor = 0;
    if(randomize) Collections.shuffle(documents, new Random(System.nanoTime()));
    docIt = documents.iterator();
    startTime = System.currentTimeMillis();
  }
  
  protected boolean hasNextDocument() {
    return docIt != null && docIt.hasNext();
  }
  
  private boolean reachedEnd() {
    return cursor >= numExamples;
  }

  public int numExamples() {
    return numExamples;
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
      exampleSize = Math.max(exampleSize, example.countSentences());
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
  public MultiDataSet next() {
    return next(batchSize);
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
    reportProgress();
    log.trace("Iterate: example size " + batch.size + " Documents x " + batch.maxDocLength + " Sentences");
    return batch;
  }
  
  @Override
  public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
  }

  @Override
  public MultiDataSetPreProcessor getPreProcessor() {
    return null;
  }

  protected void reportProgress() {
    if(stage.equals(Stage.ENCODE)) return;
    String timeStr = "??";
    try {
      long elapsed = System.currentTimeMillis() - startTime;
      long expected = elapsed * numExamples / cursor;
      long remaining = expected - elapsed;
      timeStr = String.format("%02d:%02d:%02d",
              TimeUnit.MILLISECONDS.toHours(remaining),
              TimeUnit.MILLISECONDS.toMinutes(remaining) -  
              TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(remaining)),
              TimeUnit.MILLISECONDS.toSeconds(remaining) - 
              TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(remaining)));   
    } catch(Exception e) {
    }
		int progress = (int) ((float) cursor * 100 / numExamples);
		log.debug(stage.toString() + ": returning " + cursor + "/" + numExamples + " examples [" + progress + "%, " + timeStr + " remaining]");
	}
  
  public abstract MultiDataSet generateDataSet(DocumentBatch batch);
  
}
