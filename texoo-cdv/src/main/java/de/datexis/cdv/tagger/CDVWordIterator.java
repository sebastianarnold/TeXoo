package de.datexis.cdv.tagger;

import com.google.common.collect.Lists;
import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.index.EntityIndex;
import de.datexis.cdv.model.AspectAnnotation;
import de.datexis.cdv.model.EntityAnnotation;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncodingHelpers;
import de.datexis.encoder.IEncoder;
import de.datexis.encoder.impl.DummyEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Iterates through a Dataset with Document-Level Batches of Sentences
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class CDVWordIterator extends CDVSentenceIterator {
  
  protected int maxWordsPerSentence;

  public CDVWordIterator(Stage stage, Collection<Document> docs, CDVTagger tagger, int numExamples, int maxSentencesPerDoc, int maxWordsPerSentence, int batchSize, boolean randomize, boolean balancing) {
    super(stage, docs, tagger, numExamples, maxSentencesPerDoc, batchSize, randomize, balancing);
    this.maxWordsPerSentence = maxWordsPerSentence;
    reset();
  }
  
  /** used for unit testing */
  protected CDVWordIterator(Stage stage, Collection<Document> docs, IEncoder inputEncoder, int batchSize) {
    super(stage, docs, new CDVTagger(), -1, -1, batchSize, false, false);
    tagger.setInputEncoders(inputEncoder, new DummyEncoder());
    tagger.setAspectEncoder(new DummyEncoder());
    this.maxWordsPerSentence = -1;
    reset();
  }
  
  public DocumentBatch nextDocumentBatch(int num) {
    DocumentBatch batch = nextBatch(num);
    if(this.maxWordsPerSentence > 0 && batch.maxSentenceLength > this.maxWordsPerSentence)
      batch.maxSentenceLength = this.maxWordsPerSentence;
    batch.dataset = generateDataSet(batch);
    reportProgress(batch.maxDocLength, batch.maxSentenceLength);
    return batch;
  }
  
  @Override
  public MultiDataSet generateDataSet(DocumentBatch batch) {
    // return all encodings on Sentence level
    //  4D: [ batchSize(documents) x sentences x words x vectorSize ]
    // RNN: [ batchSize(sentences) x vectorSize x words ]
    INDArray input = Nd4j.zeros(DataType.FLOAT, batch.size * batch.maxDocLength, 1, this.maxWordsPerSentence, tagger.inputEncoder.getEmbeddingVectorSize());
    //INDArray dummyInput = Nd4j.zeros(DataType.FLOAT, 1, 1); // dummy value
    INDArray wordMask = Nd4j.zeros(DataType.FLOAT, batch.size * batch.maxDocLength, 1, this.maxWordsPerSentence, 1); // this mask is used by the word encoder LSTM and lasttimestep
    INDArray sentMask = createMask(batch.docs, batch.maxDocLength, Sentence.class);
    //INDArray sentMask = Nd4j.zeros(DataType.FLOAT, batch.size, batch.maxDocLength); // this is the mask for the entire network
    INDArray entityLabelMask = createMask(batch.docs, batch.maxDocLength, Sentence.class); // will be altered during encoding
    INDArray aspectLabelMask = createMask(batch.docs, batch.maxDocLength, Sentence.class); // will be altered during encoding
    INDArray flag = tagger.flagEncoder instanceof Encoder ?
      ((Encoder)tagger.flagEncoder).encodeMatrix(batch.docs, batch.maxDocLength, Sentence.class) : // do it all in a batch
      EncodingHelpers.encodeTimeStepMatrix(batch.docs, tagger.flagEncoder, batch.maxDocLength, Sentence.class);
  
    // input encodings
    Document example;
    INDArrayIndex[] indices = new INDArrayIndex[4];
    for(int batchIndex = 0; batchIndex < batch.size; batchIndex++) {
      indices[1] = NDArrayIndex.point(0);
      example = batch.docs.get(batchIndex);
      List<Sentence> sentences = Lists.newArrayList(example.getSentences());
      for(int s = 0; s < sentences.size() && s < batch.maxDocLength; s++) {
        indices[0] = NDArrayIndex.point(batchIndex * batchSize + s);
        Sentence sent = sentences.get(s);
        List<Token> tokens =   Lists.newArrayList(sent.getTokens());
        //sentMask.putScalar(new int[] {batchIndex, s}, 1);
        for(int t = 0; t < tokens.size() && t < batch.maxSentenceLength; t++) {
          INDArray vec = tagger.inputEncoder.encode(tokens.get(t));
          // reverse tokens per sentence
          //int tr = Math.min(batch.maxSentenceLength, tokens.size()) - t - 1;
          indices[2] = NDArrayIndex.point(t);
          indices[3] = NDArrayIndex.all();
          //input.slice(batchIndex).slice(s).slice(t).assign(vec); // should be equal
          input.put(indices, vec);
          //wordMask.putScalar(new int[] {batchIndex, s, t}, 1);
        }
        if(tokens.size() >= batch.maxSentenceLength) wordMask.slice(batchIndex * batch.size + s).assign(1.0);
        else wordMask.get(NDArrayIndex.point(batchIndex * batch.size + s), NDArrayIndex.point(0), NDArrayIndex.interval(0, tokens.size()), NDArrayIndex.point(0)).assign(1.0);
      }
    }

    // target encodings
    INDArray entityTarget = null, aspectTarget = null;
    if(stage.equals(Stage.TRAIN) || stage.equals(Stage.TEST)) {
      if(tagger.entityEncoder != null && tagger.entityEncoder instanceof EntityIndex)
        entityTarget = encodeTarget(entityLabelMask, batch.docs, batch.maxDocLength, tagger.entityEncoder, Sentence.class, EntityAnnotation.class);
      if(tagger.aspectEncoder != null && tagger.aspectEncoder instanceof AspectIndex)
        aspectTarget = encodeTarget(aspectLabelMask, batch.docs, batch.maxDocLength, tagger.aspectEncoder, Sentence.class, AspectAnnotation.class);
    } else {
      // Leave empty
      if(tagger.entityEncoder != null)
        entityTarget = EncodingHelpers.createTimeStepMatrix(batch.size, tagger.entityEncoder.getEmbeddingVectorSize(), batch.maxDocLength);
      if(tagger.aspectEncoder != null)
        aspectTarget = EncodingHelpers.createTimeStepMatrix(batch.size, tagger.aspectEncoder.getEmbeddingVectorSize(), batch.maxDocLength);
    }
  
    if(entityTarget != null && aspectTarget != null) {
      // Multi-Task model
      return new org.nd4j.linalg.dataset.MultiDataSet(
        new INDArray[]{input, flag},
        new INDArray[]{entityTarget, aspectTarget},
        new INDArray[]{wordMask, sentMask},
        new INDArray[]{entityLabelMask, aspectLabelMask}
      );
    } else {
      // Single-Task model
      return new org.nd4j.linalg.dataset.MultiDataSet(
        new INDArray[]{input, flag},
        new INDArray[]{entityTarget != null ? entityTarget : aspectTarget},
        new INDArray[]{wordMask, sentMask},
        new INDArray[]{entityTarget != null ? entityLabelMask : aspectLabelMask}
      );
    }
  }
  
  protected void reportProgress(int docLength, int sentLength) {
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
    log.debug("{}: returning {}/{} examples in [{}%, {} remaining] [{} x {}]", stage.toString(), cursor, numExamples, progress, timeStr, docLength, sentLength);
  }

}
