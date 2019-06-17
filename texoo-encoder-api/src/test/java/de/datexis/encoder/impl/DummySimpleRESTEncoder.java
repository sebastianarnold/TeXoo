package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.Collection;

public class DummySimpleRESTEncoder extends SimpleRESTEncoder {
  public DummySimpleRESTEncoder(RESTAdapter restAdapter, Class<? extends Span> elementClass) {
    super(restAdapter, elementClass);
  }

  public DummySimpleRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier, Class<? extends Span> elementClass) {
    super(restAdapter, vectorIdentifier, elementClass);
  }

  @Override
  public INDArray encodeImpl(String word) throws IOException {
    return null;
  }

  @Override
  public INDArray encodeImpl(Span span) throws IOException {
    return null;
  }

  @Override
  public void encodeEachImpl(Sentence input) throws IOException {

  }

  @Override
  public void encodeEachImpl(Document input) throws IOException {

  }

  @Override
  public void encodeEachImpl(Collection<Document> docs) throws IOException {

  }
}
