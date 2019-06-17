package de.datexis.encoder.impl;

import de.datexis.encoder.impl.serde.DeserializationProvider;
import de.datexis.encoder.impl.serde.JacksonSerdeProvider;
import de.datexis.encoder.impl.serde.SerializationProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

public class SectorRESTAdapter extends AbstractRESTAdapter {

  private static final Logger log = LoggerFactory.getLogger(SectorRESTAdapter.class);

  public static final int DEFAULT_READ_TIMEOUT = 300000;
  public static final int DEFAULT_CONNECT_TIMEOUT = 10000;
  public static final long DEFAULT_EMBEDDING_VECTOR_SIZE = 128;

  public static final String URL_FORMAT = "http://%s:%d/v2/%s";

  public static final String SENTENCE_ENDPOINT = "encode/sentence";
  public static final String DOCUMENT_ENDPOINT = "encode/document";

  private String domain;
  private int port;

  private JacksonSerdeProvider jacksonSerdeProvider;

  public SectorRESTAdapter(String domain, int port, long embeddingVectorSize, int connectTimeout, int readTimeout) {
    super(embeddingVectorSize, connectTimeout, readTimeout);
    this.domain = domain;
    this.port = port;

    jacksonSerdeProvider = new JacksonSerdeProvider();
  }

  public SectorRESTAdapter(String domain, int port) {
    this(domain, port, DEFAULT_EMBEDDING_VECTOR_SIZE, DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT);
  }

  @Override
  public double[] encodeImpl(String data) throws IOException {
    return request(data, double[].class, getUrl(SENTENCE_ENDPOINT));
  }

  @Override
  public double[][] encodeImpl(String[] data) throws IOException {
    return request(data, double[][].class, getUrl(DOCUMENT_ENDPOINT));
  }

  @Override
  public double[][][] encodeImpl(String[][] data) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public SerializationProvider getSerializationProvider() {
    return jacksonSerdeProvider;
  }

  @Override
  public DeserializationProvider getDeserializationProvider() {
    return jacksonSerdeProvider;
  }

  private URL getUrl(String path) throws MalformedURLException {
    return new URL(String.format(URL_FORMAT, domain, port, path));
  }
}
