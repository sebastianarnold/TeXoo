package de.datexis.ner.tagger;

import de.datexis.encoder.EncoderSet;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.model.tag.Tag;
import de.datexis.tagger.CachedSentenceIterator;
import edu.stanford.nlp.util.Iterables;
import java.util.ArrayList;
import java.util.Collection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * Iterates through a Document, one Sentence per Example. Used for Named Entity Mentions.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MentionTaggerIterator extends CachedSentenceIterator {

  protected Annotation.Source source = Annotation.Source.GOLD;
  
  /**
   * Create an Iterator for training. Each example is a sentence.
   * @param docs the documents to use for training.
   * @param name
   * @param encoders input encoders to use. Tokens will be encoded and cached on the fly.
   * @param tagset the tagset to use for prediction.
   * @param source the Annotation source to use for prediction
   * @param numExamples number of examples to train. -1 for complete dataset.
   * @param batchSize batch size in sentences
   * @param randomize randomize examples before training epoch
   */
  public MentionTaggerIterator(Collection<Document> docs, String name, EncoderSet encoders, Class tagset, Annotation.Source source, int numExamples, int batchSize, boolean randomize) {
    super(docs, name, numExamples, batchSize, randomize);
    log = LoggerFactory.getLogger(MentionTaggerIterator.class);
    this.source = source;
    this.encoders = encoders;
    this.tagset = tagset;
    try {
      this.inputSize = encoders.getVectorSize();
      this.labelSize = this.tagset.newInstance().getVectorSize();
    } catch (InstantiationException | IllegalAccessException ex) {
      log.error("Could not instantiate target class " + tagset.getName());
    }
    reset();
  }
  
  public MentionTaggerIterator(Collection<Document> docs, String name, EncoderSet encoders, Class tagset, int numExamples, int batchSize, boolean randomize) {
    this(docs, name, encoders, tagset, Annotation.Source.GOLD, numExamples, batchSize, randomize);
  }
  
  /**
   * Returns the next Sentence in the Dataset as Tokens.
   * These Tokens contain BOD, BOS, EOS and EOD Markers.
   * @return 
   */
  public ArrayList<Token> nextTokens() {
    // Beginning of Document
    // Beginning of Sentence
    return  Iterables.asArrayList(nextSentence().getTokens().iterator());
    // End of Sentence
    // End of Document
  }

  @Override
  public DataSet generateDataSet(ArrayList<Sentence> examples, int num, int exampleSize) {
    Sentence example;
    INDArray input = Nd4j.zeros(new int[]{num, inputSize, exampleSize});
		INDArray label = Nd4j.zeros(new int[]{num, labelSize, exampleSize});
    INDArray featuresMask =  Nd4j.zeros(new int[]{num, exampleSize});
    INDArray labelsMask =  Nd4j.zeros(new int[]{num, exampleSize});
    DataSet result = new DataSet(input, label, featuresMask, labelsMask);
    for(int batchNum=0; batchNum<num; batchNum++ ) {
      //log.info("Training " + cursor + ": " + s.toString());
      example = examples.get(batchNum);
      for(int t=0; t<example.countTokens(); t++) {
        //log.trace(example.get(t).toString());
        featuresMask.put(batchNum, t, 1); // mark this word as used
        labelsMask.put(batchNum, t, 1); // mark this word as labeled
        INDArray inputEncoding = example.getToken(t).getVector(encoders);
        result.getFeatures().getRow(batchNum).getColumn(t).assign(inputEncoding);
        Tag goldLabel = example.getToken(t).getTag(source, tagset);
        result.getLabels().getRow(batchNum).getColumn(t).assign(goldLabel.getVector());
        //System.out.println(batchNum + ": " + example.getToken(t).getText() + "\t" + inputEncoding.sumNumber().toString() + "\t" + goldLabel.getVector().toString());
			}
		}
    if(clearCache()) {
      log.trace("Iterate: cleared embeddings []");
    }
    return result;
  }
  
}
