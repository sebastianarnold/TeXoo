package encoder.bert;

import com.google.gson.Gson;
import encoder.AbstractRESTAdapter;
import encoder.serialization.DeserializationProvider;
import encoder.serialization.JacksonProvider;
import encoder.serialization.SerializationProvider;
import de.datexis.model.Document;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Arrays;

public class BertRESTAdapter extends AbstractRESTAdapter {
  public static final int DEFAULT_READ_TIMEOUT = 300000;
  public static final int DEFAULT_CONNECT_TIMEOUT = 10000;
  public static final long DEFAULT_EMBEDDING_VECTOR_SIZE = 768;

  public static final String URL_FORMAT = "http://%s:%d/encode";


  private static String endpoint;
  private String domain;
  private int port;

  private JacksonProvider serdeProvider;


  public BertRESTAdapter(String domain, int port, long embeddingVectorSize) {
    super(embeddingVectorSize, DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT);
    this.domain = domain;
    this.port = port;
    serdeProvider = new JacksonProvider();

  }

  public encoder.bert.BertResponse simpleRequest(String request) throws IOException {
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

    encoder.bert.BertResponse bertResponse = null;

    try (BufferedReader br = new BufferedReader(
      new InputStreamReader(con.getInputStream(), "utf-8"))) {
      StringBuilder response = new StringBuilder();
      String responseLine = null;
      while ((responseLine = br.readLine()) != null) {
        response.append(responseLine);
      }

      Gson gson = new Gson();
      bertResponse = gson.fromJson(response.toString(), encoder.bert.BertResponse.class);
    }

    return bertResponse;
  }

  public encoder.bert.BertNonTokenizedResponse simpleRequestNonTokenized(Document d, int maxSequenceLength) throws IOException {
    Gson gson = new Gson();
    String[] sentences = new String[Math.min(d.getSentences().size(),maxSequenceLength)];
    for (int i = 0; i < Math.min(d.getSentences().size(), maxSequenceLength); ++i) {
      sentences[i] = d.getSentence(i).getText();
    }
    encoder.bert.BaaSRequestNonTokenized request = new encoder.bert.BaaSRequestNonTokenized();
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

    encoder.bert.BertNonTokenizedResponse bertResponse = null;

    try (BufferedReader br = new BufferedReader(
      new InputStreamReader(con.getInputStream(), "utf-8"))) {
      StringBuilder response = new StringBuilder();
      String responseLine = null;
      while ((responseLine = br.readLine()) != null) {
        response.append(responseLine);
      }

      gson = new Gson();
      bertResponse = gson.fromJson(response.toString(), encoder.bert.BertNonTokenizedResponse.class);
    }

    return bertResponse;
  }


  @Override
  public double[] encodeImpl(String data) throws IOException {
    encoder.bert.BaaSRequest req = new encoder.bert.BaaSRequest();
    req.texts = new String[1][1];
    req.texts[0][0] = data;
    encoder.bert.BertResponse resp = request(req, encoder.bert.BertResponse.class, getUrl());
    double[][] result = new double[1][1];
    result = Arrays.copyOfRange(resp.result[0], 1, resp.result[0].length - 1);
    return result[0];
  }

  @Override
  public double[][] encodeImpl(String[] data) throws IOException {
    // encode sentence
    encoder.bert.BaaSRequest req = new encoder.bert.BaaSRequest();
    req.texts = new String[1][data.length];
    req.texts[0] = data;
    encoder.bert.BertResponse resp = request(req, encoder.bert.BertResponse.class, getUrl());
    double[][] result = new double[1][data.length];
    // response.results has two more encoded tokens due to the bert separator and cls-token
    // so we remove the first and the last row of the returned matrix
    result = Arrays.copyOfRange(resp.result[0], 1, resp.result[0].length - 1);
    return result;
  }

  @Override
  public double[][][] encodeImpl(String[][] data) throws IOException {
    // encode document
    encoder.bert.BaaSRequest req = new encoder.bert.BaaSRequest();
    req.texts = data;
    encoder.bert.BertResponse resp = request(req, encoder.bert.BertResponse.class, getUrl());
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
