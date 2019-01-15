package de.datexis.tagger;

import com.google.common.collect.FluentIterable;
import de.datexis.model.Document;
import de.datexis.model.Token;
import java.util.ArrayList;
import java.util.Arrays;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.commons.compress.utils.Lists;

/**
 *
 * @author tsla
 */
public abstract class CachedTokenIterator extends AbstractIterator {

	protected static Logger log = LoggerFactory.getLogger(CachedTokenIterator.class);

  protected Iterable<Token> tokens;
  
  /**
   * Create a new Iterator
   * @param name Name of the Dataset
   * @param tokens Entities to iterate
   * @param numExamples Number of examples that the iterator should return per epoch
   * @param batchSize Batch size of examples per step
   * @param randomize If True, the entity order will be randomized before every epoch
   */
	public CachedTokenIterator(Iterable<Token> tokens, String name, int numExamples, int batchSize, boolean randomize) {
    super(Arrays.asList(new Document()), name, numExamples, batchSize, randomize);
    this.name = name;
    this.randomize = randomize;
    this.tokens = tokens;
    this.totalExamples = FluentIterable.from(tokens).size();
    this.numExamples = numExamples < 0 ? this.totalExamples : numExamples;    
    this.batchSize = batchSize;
	}

  @Override
	public abstract void reset();
  
  @Override
	public abstract boolean hasNext();
  
	/**
   * Returns a DataSet with batchSize tokens/Sentences.
   * @return 
   */
  @Override
  public DataSet next() {
    return next(batchSize);
  }
  
	protected void reportProgress() {
    String timeStr = "??";
    try {
      long elapsed = System.currentTimeMillis() - startTime;
      long expected = elapsed * numExamples() / cursor();
      long remaining = expected - elapsed;
      timeStr = String.format("%02d:%02d:%02d",
              TimeUnit.MILLISECONDS.toHours(remaining),
              TimeUnit.MILLISECONDS.toMinutes(remaining) -  
              TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(remaining)),
              TimeUnit.MILLISECONDS.toSeconds(remaining) - 
              TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(remaining)));   
    } catch(Exception e) {
    }
		int progress = (int) ((float) cursor() * 100 / numExamples());
		log.debug("Iterate: returning " + cursor() + "/" + numExamples() + " examples [" + progress + "%, " + timeStr + " remaining]");
	}

  /**
   * Randomizes the order of tokens.
   * @param tokens
   * @return Iterable with randomized order
   */
  protected Iterator<Token> shuffleTokens(Iterable<Token> tokens) {
    log.info("Randomizing tokens in " + name + "...");
    List<Token> shuffled = Lists.newArrayList(tokens.iterator());
    Collections.shuffle(shuffled, new Random(System.nanoTime()));
    return shuffled.iterator();
  }

  @Override
  public boolean resetSupported() {
    return true;
  }
  
  @Override
  public boolean asyncSupported() {
    return false; // TODO or do we?
  }
  
	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

  /**
   * Returns the length of the feature vector for one example.
   * @return 
   */
  @Override
  public int inputColumns() {
    return (int) inputSize;
  }

  /**
   * Retuns the length of the label vector for one example.
   * @return 
   */
  @Override
  public int totalOutcomes() {
    return (int) labelSize;
  }
  
  @Override
  public long getInputSize() {
    return inputSize;
  }

  @Override
  public long getLabelSize() {
    return labelSize;
  }
  
  /**
   * Returns the number of examples this Iterator will return.
   * @return number of tokens
   */
  @Override
  public int numExamples() {
    return numExamples;
  }
  // FIXME: DL4J docs state differently
  /**
   * Returns the total number of examples in the given dataset.
   * Note that the number of examples may be set to a different size.
   * @return number of tokens
   */
  @Override
  public int totalExamples() {
    return totalExamples;
  }
  
	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented yet.");
	}
  
  @Override
  public DataSetPreProcessor getPreProcessor() {
    return null;
  }
  
	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented yet.");
	}
  
  public Class getTagset() {
    return tagset;
  }

}
