package de.datexis.encoder.impl;

import de.datexis.encoder.impl.ELMoLayerOutput;
import de.datexis.encoder.impl.ELMoRESTAdapter;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class ELMoRESTAdapterTest {
    public static final String TEST_DOMAIN = "localhost";
    public static final int TEST_PORT = 0;

    public static final String TEST_PATH = "test";

    public static final String URL_TOP = "http://localhost:0/v2/test/top";
    public static final String URL_MIDDLE = "http://localhost:0/v2/test/middle";
    public static final String URL_BOTTOM = "http://localhost:0/v2/test/bottom";
    public static final String URL_AVERAGE = "http://localhost:0/v2/test/average";
    
    private ELMoRESTAdapter elMoRESTAdapterTop;
    private ELMoRESTAdapter elMoRESTAdapterMiddle;
    private ELMoRESTAdapter elMoRESTAdapterBottom;
    private ELMoRESTAdapter elMoRESTAdapterAverage;

    @Before
    public void setup(){
        elMoRESTAdapterTop = new ELMoRESTAdapter(ELMoLayerOutput.TOP, TEST_DOMAIN, TEST_PORT);
        elMoRESTAdapterMiddle = new ELMoRESTAdapter(ELMoLayerOutput.MIDDLE, TEST_DOMAIN, TEST_PORT);
        elMoRESTAdapterBottom = new ELMoRESTAdapter(ELMoLayerOutput.BOTTOM, TEST_DOMAIN, TEST_PORT);
        elMoRESTAdapterAverage = new ELMoRESTAdapter(ELMoLayerOutput.AVERAGE, TEST_DOMAIN, TEST_PORT);
    }

    @Test
    public void getUrlTest() throws IOException {
        URL urlTop = elMoRESTAdapterTop.getUrl(TEST_PATH);
        URL urlMiddle = elMoRESTAdapterMiddle.getUrl(TEST_PATH);
        URL urlBottom = elMoRESTAdapterBottom.getUrl(TEST_PATH);
        URL urlAverage = elMoRESTAdapterAverage.getUrl(TEST_PATH);

        assertThat(urlTop.toExternalForm(), equalTo(URL_TOP));
        assertThat(urlMiddle.toExternalForm(), equalTo(URL_MIDDLE));
        assertThat(urlBottom.toExternalForm(), equalTo(URL_BOTTOM));
        assertThat(urlAverage.toExternalForm(), equalTo(URL_AVERAGE));
    }

    @Test
    public void getConnectionTopTest() throws IOException {
        HttpURLConnection connectionTop = elMoRESTAdapterTop.getConnection(TEST_PATH);
        HttpURLConnection connectionMiddle = elMoRESTAdapterMiddle.getConnection(TEST_PATH);
        HttpURLConnection connectionBottom = elMoRESTAdapterBottom.getConnection(TEST_PATH);
        HttpURLConnection connectionAverage = elMoRESTAdapterAverage.getConnection(TEST_PATH);

        assertThat(connectionTop.getURL().toExternalForm(), equalTo(URL_TOP));
        assertThat(connectionMiddle.getURL().toExternalForm(), equalTo(URL_MIDDLE));
        assertThat(connectionBottom.getURL().toExternalForm(), equalTo(URL_BOTTOM));
        assertThat(connectionAverage.getURL().toExternalForm(), equalTo(URL_AVERAGE));
    }

    @Test
    public void configureConectionTest() throws IOException {
        HttpURLConnection connectionTop = elMoRESTAdapterTop.configureConnection(TEST_PATH);
        HttpURLConnection connectionMiddle = elMoRESTAdapterMiddle.configureConnection(TEST_PATH);
        HttpURLConnection connectionBottom = elMoRESTAdapterBottom.configureConnection(TEST_PATH);
        HttpURLConnection connectionAverage = elMoRESTAdapterAverage.configureConnection(TEST_PATH);

        verifyConfiguredHttpURLConection(connectionTop, URL_TOP);
        verifyConfiguredHttpURLConection(connectionMiddle, URL_MIDDLE);
        verifyConfiguredHttpURLConection(connectionBottom, URL_BOTTOM);
        verifyConfiguredHttpURLConection(connectionAverage, URL_AVERAGE);
    }

    private void verifyConfiguredHttpURLConection(HttpURLConnection httpURLConnection, String url){
        assertThat(httpURLConnection.getURL().toExternalForm(), equalTo(url));
        assertThat(httpURLConnection.getRequestMethod(), equalTo(ELMoRESTAdapter.HTTP_REQUEST_METHOD));
        assertThat(httpURLConnection.getRequestProperty(ELMoRESTAdapter.HTTP_CONTENT_TYPE_NAME), equalTo(ELMoRESTAdapter.HTTP_CONTENT_TYPE_VALUE));
        assertThat(httpURLConnection.getDoInput(), equalTo(true));
        assertThat(httpURLConnection.getDoOutput(), equalTo(true));
    }
}
