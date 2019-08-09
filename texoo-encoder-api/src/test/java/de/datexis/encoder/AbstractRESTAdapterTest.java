package de.datexis.encoder;

import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.spy;

public class AbstractRESTAdapterTest {
  public static final String DUMMY_TEXT = "text";
  public static final String[] DUMMY_TEXT_1D = new String[]{
    "text1",
    "text2"
  };
  public static final String[][] DUMMY_TEXT_2D = new String[][]{
    {
      "text11",
      "text12"
    },
    {
      "text21",
      "text22"
    }
  };

  public static final int EMBEDDING_VECTOR_SIZE = 100;
  public static final int CONNECT_TIMEOUT = 5000;
  public static final int READ_TIMEOUT = 1000;

  public static final String DUMMY_URL = "http://localhost:0/test";

  public AbstractRESTAdapter abstractRESTAdapter;

  @Before
  public void setup(){
    abstractRESTAdapter = spy(new DummyAbstractRESTAdapter(EMBEDDING_VECTOR_SIZE, CONNECT_TIMEOUT, READ_TIMEOUT));
  }

  @Test
  public void getConnectionTest() throws IOException {
    URL url = new URL(DUMMY_URL);

    HttpURLConnection connection = abstractRESTAdapter.getConnection(url);

    assertThat(connection.getURL().toExternalForm(), equalTo(DUMMY_URL));
  }

  @Test
  public void configureConnectionTest() throws IOException {
    URL url = new URL(DUMMY_URL);

    HttpURLConnection connection = abstractRESTAdapter.configureConnection(url);

    assertThat(connection.getURL().toExternalForm(), equalTo(DUMMY_URL));
    assertThat(connection.getRequestMethod(), equalTo(AbstractRESTAdapter.HTTP_REQUEST_METHOD));
    assertThat(connection.getRequestProperty(AbstractRESTAdapter.HTTP_ACCEPT_TYPE_NAME), equalTo(de.datexis.encoder.impl.serde.DummyProvider.ACCEPT_TYPE));
    assertThat(connection.getRequestProperty(AbstractRESTAdapter.HTTP_CONTENT_TYPE_NAME), equalTo(de.datexis.encoder.impl.serde.DummyProvider.CONTENT_TYPE));
    assertThat(connection.getConnectTimeout(), equalTo(CONNECT_TIMEOUT));
    assertThat(connection.getReadTimeout(), equalTo(READ_TIMEOUT));
    assertThat(connection.getDoInput(), equalTo(true));
    assertThat(connection.getDoOutput(), equalTo(true));
  }

  public void encodeTest() throws IOException {
    double[] embedding = abstractRESTAdapter.encode(DUMMY_TEXT);
    assertThat(embedding.length, equalTo(EMBEDDING_VECTOR_SIZE));
  }

  public void encodeTest1D() throws IOException {
    double[][] embedding = abstractRESTAdapter.encode(DUMMY_TEXT_1D);

    assertThat(embedding.length, equalTo(DUMMY_TEXT_1D.length));

    for (int i = 0; i < embedding.length; i++) {
      assertThat(embedding[i].length, equalTo(EMBEDDING_VECTOR_SIZE));
    }
  }

  public void encodeTest2D() throws IOException {
    double[][][] embedding = abstractRESTAdapter.encode(DUMMY_TEXT_2D);

    assertThat(embedding.length, equalTo(DUMMY_TEXT_2D.length));

    for (int i = 0; i < embedding.length; i++) {
      assertThat(embedding[i].length, equalTo(DUMMY_TEXT_2D[i].length));
      for(int n = 0; n < embedding[i].length; n++) {
        assertThat(embedding[i][n].length, equalTo(EMBEDDING_VECTOR_SIZE));
      }

    }
  }
}
