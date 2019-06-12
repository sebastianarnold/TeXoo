package de.datexis.encoder.impl;

import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.net.URL;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class SkipthoughtRestAdapterTest {
    public static final String TEST_DOMAIN = "localhost";
    public static final int TEST_PORT = 0;

    public static final String TEST_PATH = "test";

    public static final String FINAL_URL = "http://localhost:0/v2/test";

    private SkipthoughtRESTAdapter skipthoughtRESTAdapter;

    @Before
    public void setup(){
        skipthoughtRESTAdapter = new SkipthoughtRESTAdapter(TEST_DOMAIN, TEST_PORT);
    }

    @Test
    public void getUrlTest() throws IOException {
        URL url = skipthoughtRESTAdapter.getUrl(TEST_PATH);

        assertThat(url.toExternalForm(), equalTo(FINAL_URL));
    }
}
