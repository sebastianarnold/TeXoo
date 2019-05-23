package de.datexis.encoder.impl;

import java.io.IOException;

public interface RESTAdapter {
  double[] encode(String data) throws IOException;

  double[][] encode(String[] data) throws IOException;

  double[][][] encode(String[][] data) throws IOException;
}
