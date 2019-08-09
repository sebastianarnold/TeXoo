package de.datexis.encoder;

import encoder.AbstractRESTAdapter;
import encoder.serialization.DeserializationProvider;
import encoder.serialization.SerializationProvider;

import java.io.IOException;

public class DummyAbstractRESTAdapter extends AbstractRESTAdapter {
  private de.datexis.encoder.impl.serde.DummyProvider dummyProvider;

  public DummyAbstractRESTAdapter(long embeddingVectorSize, int connectTimeout, int readTimeout) {
    super(embeddingVectorSize, connectTimeout, readTimeout);

    dummyProvider = new de.datexis.encoder.impl.serde.DummyProvider();
  }

  @Override
  public double[] encodeImpl(String data) throws IOException {
    return new double[(int)this.getEmbeddingVectorSize()];
  }

  @Override
  public double[][] encodeImpl(String[] data) throws IOException {
    return new double[data.length][(int)this.getEmbeddingVectorSize()];
  }

  @Override
  public double[][][] encodeImpl(String[][] data) throws IOException {
    return new double[data.length][data[0].length][(int)this.getEmbeddingVectorSize()];
  }

  @Override
  public SerializationProvider getSerializationProvider() {
    return dummyProvider;
  }

  @Override
  public DeserializationProvider getDeserializationProvider() {
    return dummyProvider;
  }
}
