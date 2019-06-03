package de.datexis.encoder.impl;

import de.datexis.encoder.impl.serde.DeserializationProvider;
import de.datexis.encoder.impl.serde.SerializationProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

public abstract class AbstractRESTAdapter implements RESTAdapter {
  private static final Logger log = LoggerFactory.getLogger(AbstractRESTAdapter.class);

  public static final String HTTP_REQUEST_METHOD = "POST";
  public static final String HTTP_CONTENT_TYPE_NAME = "Content-Type";
  public static final String HTTP_ACCEPT_TYPE_NAME = "Accept";

  private long embeddingVectorSize;

  private int connectTimeout;
  private int readTimeout;

  public AbstractRESTAdapter(long embeddingVectorSize, int connectTimeout, int readTimeout) {
    this.connectTimeout = connectTimeout;
    this.readTimeout = readTimeout;
    this.embeddingVectorSize = embeddingVectorSize;
  }

  @Override
  public long getEmbeddingVectorSize() {
    return embeddingVectorSize;
  }

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

  /*
  log.debug("connect to: {}", httpConnection.getURL());
    httpConnection.connect();

    log.debug("writing to: {}", httpConnection.getURL());
    writeRequestBody(data, httpConnection);

    log.debug("reading from: {}", httpConnection.getURL());
    O responseData = readResponseData(httpConnection, classOfO);
    log.debug("response read from: {}", httpConnection.getURL());
   */

  public <I,O> O request(I input, Class<O> classOfO, URL url) throws IOException{
    log.debug("building request");
    HttpURLConnection httpConnection = configureConnection(url);

    log.debug("connect to: {}", httpConnection.getURL());
    httpConnection.connect();

    log.debug("writing to: {}", httpConnection.getURL());
    getSerializationProvider().serialize(input, httpConnection.getOutputStream());

    log.debug("reading from: {}", httpConnection.getURL());
    O output = getDeserializationProvider().deserialize(httpConnection.getInputStream(), classOfO);
    log.debug("response read from: {}", httpConnection.getURL());

    httpConnection.disconnect();
    log.debug("disconnected from: {}", httpConnection.getURL());

    return output;
  }

  public HttpURLConnection configureConnection(URL url) throws IOException{
    HttpURLConnection httpConnection = getConnection(url);
    httpConnection.setRequestMethod(HTTP_REQUEST_METHOD);
    httpConnection.setRequestProperty(HTTP_CONTENT_TYPE_NAME, getSerializationProvider().getContentType());
    httpConnection.setRequestProperty(HTTP_ACCEPT_TYPE_NAME, getDeserializationProvider().getAcceptType());
    httpConnection.setConnectTimeout(connectTimeout);
    httpConnection.setReadTimeout(readTimeout);
    httpConnection.setDoOutput(true);
    httpConnection.setDoInput(true);

    return httpConnection;
  }

  public abstract SerializationProvider getSerializationProvider();

  public abstract DeserializationProvider getDeserializationProvider();

  public HttpURLConnection getConnection(URL url) throws IOException {
    return (HttpURLConnection) url.openConnection();
  }


}
