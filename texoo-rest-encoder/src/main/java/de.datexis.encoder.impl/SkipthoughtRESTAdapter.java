package de.datexis.encoder.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public class SkipthoughtRESTAdapter extends AbstractRESTAdapter {
  private static final Logger log = LoggerFactory.getLogger(SkipthoughtRESTAdapter.class);

  public static final String URL_FORMAT = "http://%s:%d/v2/%s";

  public static final String SENTENCE_ENDPOINT = "embed/sentences";
  public static final String SENTENCES_ENDPOINT = "embed/sentences";

  public static final String HTTP_REQUEST_METHOD = "POST";
  public static final String HTTP_CONTENT_TYPE_NAME = "Content-Type";
  public static final String HTTP_CONTENT_TYPE_VALUE = "application/json; charset=UTF-8";

  private String domain;
  private int port;

  private ObjectMapper objectMapper;

  public SkipthoughtRESTAdapter(String domain, int port) {
    this.domain = domain;
    this.port = port;

    objectMapper = new ObjectMapper();
  }

  @Override
  public double[] encodeImpl(String sentence) throws IOException{
    return request(sentence, SENTENCE_ENDPOINT, double[].class);
  }

  @Override
  public double[][] encodeImpl(String[] sentencesOfDocument) throws IOException{
    return request(sentencesOfDocument, SENTENCES_ENDPOINT, double[][].class);
  }

  @Override
  public double[][][] encodeImpl(String[][] data) throws IOException {
    throw new UnsupportedOperationException();
  }

  public<I,O> O request(I data, String path, Class<O> classOfO) throws IOException{
    HttpURLConnection httpConnection = configureConnection(path);
    httpConnection.connect();

    log.debug("connect to: {}", httpConnection.getURL());
    writeRequestBody(data, httpConnection);

    log.debug("writing to: {}", httpConnection.getURL());
    O responseData = readResponseData(httpConnection, classOfO);

    log.debug("reading from: {}", httpConnection.getURL());
    httpConnection.disconnect();
    log.debug("response read from: {}", httpConnection.getURL());

    return responseData;
  }

  public <T> void writeRequestBody(T data, HttpURLConnection httpConnection) throws IOException {
    OutputStream outputStream = httpConnection.getOutputStream();
    objectMapper.writeValue(outputStream, data);
    outputStream.close();
  }

  public <T> T readResponseData(HttpURLConnection httpURLConnection, Class<T> classOfT)
      throws IOException {
    InputStream inputStream = httpURLConnection.getInputStream();
    T responseData = objectMapper.readValue(inputStream, classOfT);
    inputStream.close();
    return responseData;
  }

  public HttpURLConnection configureConnection(String path) throws IOException{
    HttpURLConnection httpConnection = getConnection(path);
    httpConnection.setRequestMethod(HTTP_REQUEST_METHOD);
    httpConnection.setRequestProperty(HTTP_CONTENT_TYPE_NAME,HTTP_CONTENT_TYPE_VALUE);
    httpConnection.setDoOutput(true);
    httpConnection.setDoInput(true);

    return httpConnection;
  }

  public HttpURLConnection getConnection(String path) throws IOException {
    URL url = getUrl(path);
    return (HttpURLConnection) url.openConnection();
  }

  public URL getUrl(String path) throws MalformedURLException {
      return new URL(String.format(URL_FORMAT, domain, port, path));
  }
}
