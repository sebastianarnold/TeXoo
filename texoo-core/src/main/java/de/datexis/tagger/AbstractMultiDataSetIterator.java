package de.datexis.tagger;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * Base class for MultiDatasetIterators based on TeXoo Documents.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class AbstractMultiDataSetIterator implements MultiDataSetIterator {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  public static enum Stage { TRAIN, TEST, ENCODE };
  
  protected long numExamples;
  protected int batchSize = -1;
  protected int maxTimeSeriesLength = -1;
  protected int cursor;
  protected long startTime;
  protected boolean randomize;
  
  protected Stage stage;
  
  public AbstractMultiDataSetIterator(Stage stage, int numExamples, int maxTimeSeriesLength, int batchSize, boolean randomize) {
    this.numExamples = numExamples;
    this.maxTimeSeriesLength = maxTimeSeriesLength;
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
  
  public long getNumExamples() {
    return numExamples;
  }
  
  @Override
  public void reset() {
    cursor = 0;
    startTime = System.currentTimeMillis();
  }
  
  protected boolean reachedEnd() {
    return cursor >= numExamples;
  }
  
  @Override
  public MultiDataSet next() {
    return next(batchSize);
  }
  
  @Override
  public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
  }
  
  @Override
  public MultiDataSetPreProcessor getPreProcessor() {
    return null;
  }
  
  protected void reportProgress(int maxLength) {
    //if(stage.equals(Stage.TEST)) Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();
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
    // TODO: add a warning if batch length was truncated!
    log.debug("{}: returning {}/{} examples in [{}%, {} remaining] [batch length {}]", stage.toString(), cursor, numExamples, progress, timeStr, maxLength);
  }
  
}
