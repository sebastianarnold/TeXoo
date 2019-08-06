package de.datexis.encoder.impl.BERTAsAService;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import de.datexis.encoder.impl.AbstractRESTAdapter;
import de.datexis.encoder.impl.serde.DeserializationProvider;
import de.datexis.encoder.impl.serde.JacksonSerdeProvider;
import de.datexis.encoder.impl.serde.SerializationProvider;
import de.datexis.model.Document;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;

public class BertRESTAdapter extends AbstractRESTAdapter {
  public static final int DEFAULT_READ_TIMEOUT = 300000;
  public static final int DEFAULT_CONNECT_TIMEOUT = 10000;
  public static final long DEFAULT_EMBEDDING_VECTOR_SIZE = 768;

  public static final String URL_FORMAT = "http://%s:%d/encode";


  private static String endpoint;
  private String domain;
  private int port;

  private JacksonSerdeProvider serdeProvider;


  public BertRESTAdapter(String domain, int port, long embeddingVectorSize) {
    super(embeddingVectorSize, DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT);
    this.domain = domain;
    this.port = port;
    serdeProvider = new JacksonSerdeProvider();

  }

  public BertResponse simpleRequest(String request) throws IOException {
    URL url = getUrl();
    HttpURLConnection con = (HttpURLConnection) url.openConnection();
    con.setRequestMethod("POST");
    con.setRequestProperty("Content-Type", "application/json");
    con.setDoInput(true);
    con.setDoOutput(true);
    try (OutputStream os = con.getOutputStream()) {
      byte[] input = request.getBytes("utf-8");
      os.write(input, 0, input.length);
    }

    BertResponse bertResponse = null;

    try (BufferedReader br = new BufferedReader(
      new InputStreamReader(con.getInputStream(), "utf-8"))) {
      StringBuilder response = new StringBuilder();
      String responseLine = null;
      while ((responseLine = br.readLine()) != null) {
        response.append(responseLine);
      }

      Gson gson = new Gson();
      bertResponse = gson.fromJson(response.toString(), BertResponse.class);
    }

    return bertResponse;
  }

  public BertNonTokenizedResponse simpleRequestNonTokenized(Document d, int maxSequenceLength) throws IOException {
    Gson gson = new Gson();
    String[] sentences = new String[d.getSentences().size()];
    for (int i = 0; i < Math.min(d.getSentences().size(), maxSequenceLength); ++i) {
      sentences[i] = d.getSentence(i).getText();
    }
    BaaSRequestNonTokenized request = new BaaSRequestNonTokenized();
    request.texts = sentences;
    String req = gson.toJson(request);


    URL url = getUrl();
    HttpURLConnection con = (HttpURLConnection) url.openConnection();
    con.setRequestMethod("POST");
    con.setRequestProperty("Content-Type", "application/json");
    con.setDoInput(true);
    con.setDoOutput(true);
    try (OutputStream os = con.getOutputStream()) {
      byte[] input = req.getBytes("utf-8");
      os.write(input, 0, input.length);
    }

    BertNonTokenizedResponse bertResponse = null;

    try (BufferedReader br = new BufferedReader(
      new InputStreamReader(con.getInputStream(), "utf-8"))) {
      StringBuilder response = new StringBuilder();
      String responseLine = null;
      while ((responseLine = br.readLine()) != null) {
        response.append(responseLine);
      }

      gson = new Gson();
      bertResponse = gson.fromJson(response.toString(), BertNonTokenizedResponse.class);
    }

    return bertResponse;
  }


  @Override
  public double[] encodeImpl(String data) throws IOException {
    BaaSRequest req = new BaaSRequest();
    req.texts = new String[1][1];
    req.texts[0][0] = data;
    BertResponse resp = request(req, BertResponse.class, getUrl());
    double[][] result = new double[1][1];
    result = Arrays.copyOfRange(resp.result[0], 1, resp.result[0].length - 1);
    return result[0];
  }

  @Override
  public double[][] encodeImpl(String[] data) throws IOException {
    // encode sentence
    BaaSRequest req = new BaaSRequest();
    req.texts = new String[1][data.length];
    req.texts[0] = data;
    BertResponse resp = request(req, BertResponse.class, getUrl());
    double[][] result = new double[1][data.length];
    // response.results has two more encoded tokens due to the bert separator and cls-token
    // so we remove the first and the last row of the returned matrix
    result = Arrays.copyOfRange(resp.result[0], 1, resp.result[0].length - 1);
    return result;
  }

  @Override
  public double[][][] encodeImpl(String[][] data) throws IOException {
    // encode document
    BaaSRequest req = new BaaSRequest();
    req.texts = data;
    BertResponse resp = request(req, BertResponse.class, getUrl());
    double[][][] result = new double[data.length][][];
    // [sentence][token][certain EmbeddingValue]
    for (int i = 0; i < data.length; ++i) {
      result[i] = Arrays.copyOfRange(resp.result[i], 1, resp.result[i].length - 1);
    }
    return result;
  }


  @Override
  public SerializationProvider getSerializationProvider() {
    return serdeProvider;
  }

  @Override
  public DeserializationProvider getDeserializationProvider() {
    return serdeProvider;
  }

  public URL getUrl() throws MalformedURLException {
    return new URL(String.format(URL_FORMAT, domain, port));
  }
}
