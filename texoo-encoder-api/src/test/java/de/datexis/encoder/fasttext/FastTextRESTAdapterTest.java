package de.datexis.encoder.fasttext;

import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
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
}
