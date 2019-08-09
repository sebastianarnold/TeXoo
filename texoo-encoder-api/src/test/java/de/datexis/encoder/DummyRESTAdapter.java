package de.datexis.encoder;

import java.io.IOException;

public class DummyRESTAdapter implements RESTAdapter {
  private long embeddingVectorSize;

  public DummyRESTAdapter(long embeddingVectorSize) {
    this.embeddingVectorSize = embeddingVectorSize;
  }

  @Override
  public long getEmbeddingVectorSize() {
    return embeddingVectorSize;
  }

  @Override
  public double[] encode(String data) throws IOException {
    return new double[(int)embeddingVectorSize];
  }

  @Override
  public double[][] encode(String[] data) throws IOException {
    return new double[data.length][(int)embeddingVectorSize];
  }

  @Override
  public double[][][] encode(String[][] data) throws IOException {
    return new double[data.length][data[0].length][(int)embeddingVectorSize];
  }
}
