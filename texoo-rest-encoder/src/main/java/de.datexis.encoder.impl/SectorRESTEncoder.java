package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;

public class SectorRESTEncoder extends SimpleRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(SectorRESTEncoder.class);

  public static SectorRESTEncoder create(String domain, int port) {
    return new SectorRESTEncoder(new SectorRESTAdapter(domain, port));
  }

  public static SectorRESTEncoder create(String domain, int port, String vectorIdentifier) {
    return new SectorRESTEncoder(new SectorRESTAdapter(domain, port), vectorIdentifier);
  }

  public SectorRESTEncoder(RESTAdapter restAdapter) {
    super(restAdapter, Sentence.class);
  }

  public SectorRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super(restAdapter, vectorIdentifier, Sentence.class);
  }

  @Override
  public INDArray encodeImpl(String word) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public INDArray encodeImpl(Span span) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void encodeEachImpl(Sentence input) throws IOException {
    encodeEach(input, Sentence::toTokenizedString);
  }

  @Override
  public void encodeEachImpl(Document input) throws IOException {
    encodeEach1D(input.getSentences(), Sentence::toTokenizedString);
  }

  @Override
  public void encodeEachImpl(Collection<Document> docs) throws IOException {
    for (Document document : docs) {
      encodeEachImpl(document);
    }
  }
}
