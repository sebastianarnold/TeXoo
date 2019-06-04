package de.datexis.retrieval.index;

import de.datexis.common.Resource;
import de.datexis.retrieval.preprocess.WikipediaIndex;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class WikipediaIndexTest {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected final static Resource pagesSql = Resource.fromJAR("testdata/enwiki-page.sql");
  protected final static Resource redirectSql = Resource.fromJAR("testdata/enwiki-redirect.sql");
  
  public WikipediaIndexTest() {
  }
  
  @Test
  public void testReadPages() throws IOException {
    WikipediaIndex index = new WikipediaIndex();
    index.readPages(pagesSql);
    //assertEquals("AccessibleComputing", index.getTitleForId(10)); // this is a redirect, though
    Assert.assertEquals(null, index.getTitleForId(10)); // we skip redirects
    Assert.assertEquals(null, index.getTitleForId(590)); // we skip disambiguation pages
    Assert.assertEquals("Anarchism", index.getTitleForId(12));
    Assert.assertEquals("MessagePad", index.getTitleForId(887));
    Assert.assertEquals(null, index.getTitleForId(-1));
    Assert.assertEquals(null, index.getTitleForId(0));
    Assert.assertEquals("Job's_Cove", index.getTitleForId(2363508));
    Assert.assertEquals(151, index.countPages());
  }
  
  @Test
  public void testReadRedirects() throws IOException {
    WikipediaIndex index = new WikipediaIndex();
    index.readPages(pagesSql);
    index.readRedirects(redirectSql);
    Assert.assertEquals(null, index.getTitleForId(10)); // we skip redirects
    Assert.assertEquals("Computer_accessibility", index.getTitleForId(411964));
    Assert.assertEquals("Computer_accessibility", index.getTitleFromRedirect("AccessibleComputing"));
    Assert.assertEquals("Computer_accessibility", index.getTitleFromRedirect("Accessible_computing"));
    Assert.assertEquals("Computer_accessibility", index.getTitleFromRedirect("Computer_accessibility"));
    assertEquals(411964, (long) index.getIdForTitle("Computer_accessibility"));
    assertEquals(411964, (long) index.getIdForTitle("AccessibleComputing"));
    Assert.assertEquals("Anarchism", index.getTitleForId(12));
    Assert.assertEquals("Anarchism", index.getTitleFromRedirect("Anarchism"));
    Assert.assertEquals("MessagePad", index.getTitleForId(887));
    Assert.assertEquals(null, index.getTitleForId(-1));
    Assert.assertEquals(null, index.getTitleForId(0));
    Assert.assertEquals("Job's_Cove", index.getTitleForId(2363508));
  }
  
}
