package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.List;

public class SkipthoughtRESTEncoder extends AbstractRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(SkipthoughtRESTEncoder.class);

  public static SkipthoughtRESTEncoder create(String domain, int port) {
    return new SkipthoughtRESTEncoder(new SkipthoughtRESTAdapter(domain, port));
  }

  public static SkipthoughtRESTEncoder create(String domain, int port, String vectorIdentifier) {
    return new SkipthoughtRESTEncoder(new SkipthoughtRESTAdapter(domain, port), vectorIdentifier);
  }

  public SkipthoughtRESTEncoder(RESTAdapter restAdapter) {
    super(restAdapter);
  }

  public SkipthoughtRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super(restAdapter, vectorIdentifier);
  }

  /*@Override
  public long getEmbeddingVectorSize() {
    return 4800;
  }*/

  @Override
  public INDArray encode(String word) {
    try {
      return encodeImpl(word);
    } catch (IOException e) {
      log.error("IO Error while encoding word: {}", word, e);
      throw new UncheckedIOException(e);
    }
  }

  @Override
  public INDArray encode(Span span) {
    if (span instanceof Sentence) {
      return encode(span.getText());
    } else {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    if(elementClass == Sentence.class){
      try {
        /*double[] embedding = skipthoughtRESTAdapter.encode(input.getText());

        putVectorInSentence(input, embedding);*/
        encodeEach(input);
      } catch (IOException e) {
        log.error("IO Error while encoding sentence: {}", input, e);
        throw new UncheckedIOException(e);
      }
    }else{
      throw new UnsupportedOperationException();
    }
  }


  @Override
  public void encodeEach(Document input, Class<? extends Span> elementClass) {
    if (elementClass == Sentence.class) {
      try {
        /*String[] sentencesOfDocumentAsStringArray = getSentencesOfDocumentAsStringArray(input);

        double[][] embedding = skipthoughtRESTAdapter.encode(sentencesOfDocumentAsStringArray);

        putVectorInSentenceOfDocument(input, embedding);*/
        encodeEach1D(input.getSentences());
      } catch (IOException e) {
        log.error("IO Error while encoding document: {}", input.getTitle(), e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public void encodeEach(Collection<Document> docs, Class<? extends Span> elementClass) {
    for (Document document : docs) {
      encodeEach(document, elementClass);
    }
  }

  @Override
  public INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    throw new UnsupportedOperationException();
  }
}
