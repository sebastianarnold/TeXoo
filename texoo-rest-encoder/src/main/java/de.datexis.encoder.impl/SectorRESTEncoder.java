package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.List;

public class SectorRESTEncoder extends AbstractRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(SectorRESTEncoder.class);

  public static SectorRESTEncoder create(String domain, int port) {
    return new SectorRESTEncoder(new SectorRESTAdapter(domain, port));
  }

  public static SectorRESTEncoder create(String domain, int port, String vectorIdentifier) {
    return new SectorRESTEncoder(new SectorRESTAdapter(domain, port), vectorIdentifier);
  }

  public SectorRESTEncoder(RESTAdapter restAdapter) {
    super(restAdapter);
  }

  public SectorRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super(restAdapter, vectorIdentifier);
  }

  @Override
  public INDArray encode(String word) {
    throw new UnsupportedOperationException();
  }

  @Override
  public INDArray encode(Span span) {
    throw new UnsupportedOperationException();
  }

  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    if (elementClass == Sentence.class) {
      try {
        /*double[] embedding = sectorRestAdapter.encode(input.toTokenizedString());

        putVectorInSentence(input, embedding);*/
        encodeEach(input, Sentence::toTokenizedString);
      } catch (IOException e) {
        log.error("IO Error while encoding sentence: {}", input, e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException("Sector can not encode anything else than Sentences");
    }
  }

  @Override
  public void encodeEach(Document input, Class<? extends Span> elementClass) {
    if (elementClass == Sentence.class) {
      try {
        /*String[] sentencesOfDocumentAsStringArray = getSentencesOfDocumentAsStringArray(input);

        double[][] embedding = sectorRestAdapter.encode(sentencesOfDocumentAsStringArray);

        putVectorInSentenceOfDocument(input, embedding);*/
        encodeEach1D(input.getSentences(), Sentence::toTokenizedString);
      } catch (IOException e) {
        log.error("IO Error while encoding document: {}", input, e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException("Sector can not encode anything else than Sentences");
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
