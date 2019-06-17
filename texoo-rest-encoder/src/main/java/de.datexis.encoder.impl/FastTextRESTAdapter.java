package de.datexis.encoder.impl;

import de.datexis.encoder.impl.serde.DeserializationProvider;
import de.datexis.encoder.impl.serde.JacksonSerdeProvider;
import de.datexis.encoder.impl.serde.SerializationProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

public class FastTextRESTAdapter extends AbstractRESTAdapter {
  private static final Logger log = LoggerFactory.getLogger(FastTextRESTAdapter.class);

  public static final int DEFAULT_READ_TIMEOUT = 300000;
  public static final int DEFAULT_CONNECT_TIMEOUT = 10000;
  public static final long DEFAULT_EMBEDDING_VECTOR_SIZE = 300;

  public static final String URL_FORMAT = "http://%s:%d/v2/%s";

  public static final String SENTENCES_ENDPOINT = "embed/sentences";

  private String domain;
  private int port;
  private JacksonSerdeProvider serdeProvider;

  public FastTextRESTAdapter(String domain, int port, long embeddingVectorSize, int connectTimeout, int readTimeout) {
    super(embeddingVectorSize, connectTimeout, readTimeout);
    this.domain = domain;
    this.port = port;

    serdeProvider = new JacksonSerdeProvider();
  }

  public FastTextRESTAdapter(String domain, int port) {
    this(domain, port, DEFAULT_EMBEDDING_VECTOR_SIZE, DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT);
  }

  @Override
  public double[] encodeImpl(String data) throws IOException {
    throw new UnsupportedOperationException("FastText can't embed just one Token");
  }

  @Override
  public double[][] encodeImpl(String[] data) throws IOException {
    return encodeImpl(new String[][]{data})[0];
  }

  @Override
  public double[][][] encodeImpl(String[][] tokensOfDocument2D) throws IOException{
    return request(tokensOfDocument2D, double[][][].class, getUrl(SENTENCES_ENDPOINT));
  }

  @Override
  public SerializationProvider getSerializationProvider() {
    return serdeProvider;
  }

  @Override
  public DeserializationProvider getDeserializationProvider() {
    return serdeProvider;
  }

  public URL getUrl(String path) throws MalformedURLException {
    return new URL(String.format(URL_FORMAT, domain, port, path));
  }
}
