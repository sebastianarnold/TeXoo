package de.datexis.encoder.impl;

import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

public interface RESTAdapter {
  long getEmbeddingVectorSize();

  double[] encode(String data) throws IOException;

  double[][] encode(String[] data) throws IOException;

  double[][][] encode(String[][] data) throws IOException;
}
