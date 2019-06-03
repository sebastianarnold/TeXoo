package de.datexis.encoder.impl;

import de.datexis.encoder.impl.FastTextRESTAdapter;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class FastTextRESTAdapterTest {
    public static final String TEST_DOMAIN = "localhost";
    public static final int TEST_PORT = 0;

    public static final String TEST_PATH = "test";

    public static final String FINAL_URL = "http://localhost:0/v2/test";

    private FastTextRESTAdapter fastTextRESTAdapter;

    @Before
    public void setup(){
        fastTextRESTAdapter = new FastTextRESTAdapter(TEST_DOMAIN, TEST_PORT);
    }

    @Test
    public void getUrlTest() throws IOException {
        URL url = fastTextRESTAdapter.getUrl(TEST_PATH);

        assertThat(url.toExternalForm(), equalTo(FINAL_URL));
    }

    /*@Test
    public void getConnectionTest() throws IOException {
        HttpURLConnection httpURLConnection = fastTextRESTAdapter.getConnection(TEST_PATH);

        assertThat(httpURLConnection.getURL().toExternalForm(), equalTo(FINAL_URL));
    }*/

    /*@Test
    public void configureConnectionTest() throws IOException {
        HttpURLConnection httpURLConnection = fastTextRESTAdapter.configureConnection(TEST_PATH);

        assertThat(httpURLConnection.getURL().toExternalForm(), equalTo(FINAL_URL));
        assertThat(httpURLConnection.getRequestMethod(), equalTo(FastTextRESTAdapter.HTTP_REQUEST_METHOD));
        assertThat(httpURLConnection.getRequestProperty(FastTextRESTAdapter.HTTP_CONTENT_TYPE_NAME), equalTo(FastTextRESTAdapter.HTTP_CONTENT_TYPE_VALUE));
        assertThat(httpURLConnection.getDoOutput(), equalTo(true));
        assertThat(httpURLConnection.getDoInput(), equalTo(true));
    }*/
}
