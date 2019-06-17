package de.datexis.encoder.impl;

import de.datexis.encoder.impl.serde.DeserializationProvider;
import de.datexis.encoder.impl.serde.DummySerdeProvider;
import de.datexis.encoder.impl.serde.SerializationProvider;

import java.io.IOException;

public class DummyAbstractRESTAdapter extends AbstractRESTAdapter{
  private DummySerdeProvider dummySerdeProvider;

  public DummyAbstractRESTAdapter(long embeddingVectorSize, int connectTimeout, int readTimeout) {
    super(embeddingVectorSize, connectTimeout, readTimeout);

    dummySerdeProvider = new DummySerdeProvider();
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
    return dummySerdeProvider;
  }

  @Override
  public DeserializationProvider getDeserializationProvider() {
    return dummySerdeProvider;
  }
}
