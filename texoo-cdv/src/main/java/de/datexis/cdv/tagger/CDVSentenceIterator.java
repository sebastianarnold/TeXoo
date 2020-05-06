package de.datexis.cdv.tagger;

import de.datexis.cdv.CDVAnnotator;
import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.index.EntityIndex;
import de.datexis.cdv.index.QueryIndex;
import de.datexis.cdv.model.AspectAnnotation;
import de.datexis.cdv.model.EntityAnnotation;
import de.datexis.common.AnnotationHelpers;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncodingHelpers;
import de.datexis.encoder.IEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.tagger.DocumentSentenceIterator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Iterates through a Dataset with Document-Level Batches of Sentences
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class CDVSentenceIterator extends DocumentSentenceIterator {

  protected CDVTagger tagger;
  
  protected boolean balancing = true;

  public CDVSentenceIterator(Stage stage, Collection<Document> docs, CDVTagger tagger, int numExamples, int maxTimeSeriesLength, int batchSize, boolean randomize, boolean balancing) {
    super(stage, docs, numExamples, maxTimeSeriesLength, batchSize, randomize);
    this.tagger = tagger;
    this.balancing = balancing;
    reset();
  }
  
  @Override
  public MultiDataSet generateDataSet(DocumentBatch batch) {
    
    // input encodings
    INDArray inputMask = createMask(batch.docs, batch.maxDocLength, Sentence.class);
    INDArray entityLabelMask = createMask(batch.docs, batch.maxDocLength, Sentence.class); // will be altered during encoding
    INDArray aspectLabelMask = createMask(batch.docs, batch.maxDocLength, Sentence.class); // will be altered during encoding

    // return all encodings on Sentence level
    INDArray emb = tagger.inputEncoder instanceof Encoder ?
      ((Encoder)tagger.inputEncoder).encodeMatrix(batch.docs, batch.maxDocLength, Sentence.class) : // do it all in a batch
      EncodingHelpers.encodeTimeStepMatrix(batch.docs, tagger.inputEncoder, batch.maxDocLength, Sentence.class);
    INDArray flag = tagger.flagEncoder instanceof Encoder ?
      ((Encoder)tagger.flagEncoder).encodeMatrix(batch.docs, batch.maxDocLength, Sentence.class) : // do it all in a batch
      EncodingHelpers.encodeTimeStepMatrix(batch.docs, tagger.flagEncoder, batch.maxDocLength, Sentence.class);

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
        new INDArray[]{emb, flag},
        new INDArray[]{entityTarget, aspectTarget},
        new INDArray[]{inputMask, inputMask},
        new INDArray[]{entityLabelMask, aspectLabelMask}
      );
    } else {
      // Single-Task model
      return new org.nd4j.linalg.dataset.MultiDataSet(
        new INDArray[]{emb, flag},
        new INDArray[]{entityTarget != null ? entityTarget : aspectTarget},
        new INDArray[]{inputMask, inputMask},
        new INDArray[]{entityTarget != null ? entityLabelMask : aspectLabelMask}
      );
    }
  }
  
  public INDArray createMask(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {

    INDArray mask = Nd4j.zeros(DataType.FLOAT, input.size(), maxTimeSteps);
    Document example;

    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      example = input.get(batchIndex);

      int spanCount = 0;
      if(timeStepClass == Token.class) spanCount = example.countTokens();
      else if(timeStepClass == Sentence.class) spanCount = example.countSentences();

      for(int t = 0; t < spanCount && t < maxTimeSteps; t++) {
        mask.putScalar(new int[] {batchIndex, t}, 1);
      }
      
    }
    return mask;
  }

  public <S extends Span, A extends PassageAnnotation> INDArray encodeTarget(INDArray labelMask, List<Document> input, int maxTimeSteps, IEncoder encoder, Class<S> timeStepClass, Class<A> passageClass) {

    INDArray encoding = EncodingHelpers.createTimeStepMatrix(input.size(), encoder.getEmbeddingVectorSize(), maxTimeSteps);
    Document example;

    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      example = input.get(batchIndex);
      int t = 0;
      INDArray vec = Nd4j.zeros(DataType.FLOAT, encoder.getEmbeddingVectorSize(), 1);
      double weight = 0;
      Collection<A> current, last = null;
      for(Map.Entry<S, Collection<A>> ann : AnnotationHelpers.getSpanAnnotationsMultiMap(example, timeStepClass, passageClass)) {
        if(t >= maxTimeSteps) break; // limit document length
        current = ann.getValue();
        if(!current.equals(last)) { // reencode only if annotation set changed
          Map.Entry<INDArray, Double> label = CDVAnnotator.lookupAnnotations((QueryIndex) encoder, current, balancing);
          vec = label.getKey();
          weight = label.getValue();
          last = current;
        }
        // mask targets that were not found in index, use weight for used labels
        if(vec == null) {
          labelMask.putScalar(new int[] {batchIndex, t}, 0);
        } else {
          if(balancing && weight < 1.) labelMask.putScalar(new int[] {batchIndex, t}, weight);
          EncodingHelpers.putTimeStep(encoding, batchIndex, t, vec.dup());
        }
        t++;
      }
    }
    return encoding;
  }

}
