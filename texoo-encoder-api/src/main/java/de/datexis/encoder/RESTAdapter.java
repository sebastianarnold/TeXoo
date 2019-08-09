package de.datexis.encoder;

import java.io.IOException;

public interface RESTAdapter {
  
  long getEmbeddingVectorSize();

  double[] encode(String data) throws IOException;

  double[][] encode(String[] data) throws IOException;

  double[][][] encode(String[][] data) throws IOException;
  
}
