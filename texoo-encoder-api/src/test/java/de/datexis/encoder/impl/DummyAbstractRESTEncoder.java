package de.datexis.encoder.impl;

import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DummyAbstractRESTEncoder extends AbstractRESTEncoder {
  public DummyAbstractRESTEncoder(RESTAdapter restAdapter) {
    super(restAdapter);
  }

  public DummyAbstractRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super(restAdapter, vectorIdentifier);
  }

  @Override
  public INDArray encode(String word) {
    return null;
  }

  @Override
  public INDArray encode(Span span) {
    return null;
  }
}
