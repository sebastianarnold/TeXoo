package de.datexis.encoder.elmo;

import encoder.elmo.ELMoLayerOutput;
import encoder.elmo.ELMoRESTAdapter;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
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
}
