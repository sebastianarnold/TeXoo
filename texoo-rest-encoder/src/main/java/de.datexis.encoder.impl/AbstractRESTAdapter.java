package de.datexis.encoder.impl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public abstract class AbstractRESTAdapter implements RESTAdapter {

  private static final Logger log = LoggerFactory.getLogger(AbstractRESTAdapter.class);

  @Override
  public double[] encode(String data) throws IOException {
    try {
      return encodeImpl(data);
    } catch (IOException e) {
      log.error("IO error while encoding: {}", data, e);
      throw e;
    }
  }

  public abstract double[] encodeImpl(String data) throws IOException;

  @Override
  public double[][] encode(String[] data) throws IOException {
    try {
      return encodeImpl(data);
    } catch (IOException e) {
      log.error("IO error while encoding: {}", data, e);
      throw e;
    }
  }

  public abstract double[][] encodeImpl(String[] data) throws IOException;

  @Override
  public double[][][] encode(String[][] data) throws IOException {
    try {
      return encodeImpl(data);
    } catch (IOException e) {
      log.error("IO error while encoding: {}", data, e);
      throw e;
    }
  }

  public abstract double[][][] encodeImpl(String[][] data) throws IOException;
}
