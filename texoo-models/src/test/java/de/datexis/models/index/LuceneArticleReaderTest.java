package de.datexis.models.index;


import de.datexis.common.Resource;
import de.datexis.models.index.ArticleRef;
import de.datexis.models.index.ArticleIndexFactory;
import de.datexis.models.index.impl.LuceneArticleIndex;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.Before;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LuceneArticleReaderTest {
  
  @Before
  public void init() {
  }
  
  @Test
  public void testWordQueries() throws IOException {
    Resource file = Resource.fromJAR("models/Articles_en_Wikidata_250_20170828.txt");
    LuceneArticleIndex index = ArticleIndexFactory.loadWikiDataIndex(file);
    List<ArticleRef> art;
    art = index.queryNames("Berlin", 10); // 1
    assertEquals(1, art.size());
    assertEquals("Q64", art.get(0).getId());
    art = index.queryNames("berlin", 10); // 1 Q64
    assertEquals(1, art.size());
    assertEquals("Q64", art.get(0).getId());
    art = index.queryNames("Sebastian", 10); // 1 Sebastián Piñera Q306
    assertEquals(1, art.size());
    assertEquals("Q306", art.get(0).getId());
    art = index.queryNames("Eurovision Song Contest", 10); // 1 Q276
    assertEquals(1, art.size());
    assertEquals("Q276", art.get(0).getId());
    art = index.queryNames("Ubuntu", 10); // 1 Q381
    assertEquals(1, art.size());
    assertEquals("Q381", art.get(0).getId());
    art = index.queryNames("Ubuntu Linux", 10); // 1 Q381
    assertEquals(1, art.size());
    assertEquals("Q381", art.get(0).getId());
    art = index.queryNames("Ubnutu", 10); // typo
    //assertEquals(1, art.size());
    //assertEquals("Q381", art.get(0).getId());
    art = index.queryNames("Musée", 10);
    assertEquals(2, art.size());
    art = index.queryNames("fine art", 10);
    assertEquals(0, art.size());
    art = index.queryNames("fine arts", 10);
    assertEquals(1, art.size());
    art = index.queryNames("arts fine", 10);
    assertEquals(1, art.size());
  }
    
  @Test
  public void testIDQueries() throws IOException {
    Resource file = Resource.fromJAR("models/Articles_en_Wikidata_250_20170828.txt");
    LuceneArticleIndex index = ArticleIndexFactory.loadWikiDataIndex(file);
    Optional<ArticleRef> art;

    art = index.queryWikidataID("Q109");
    assertTrue(art.isPresent());
    assertEquals("Q109", art.get().getId());
    assertEquals("https://en.wikipedia.org/wiki/February", art.get().getUrl());
    assertEquals("February", art.get().getTitle());
    art = index.queryWikidataID("Q"); // wrong ID
    assertFalse(art.isPresent());
    art = index.queryWikidataID(""); // missing ID
    assertFalse(art.isPresent());
    
    art = index.queryWikipediaURL("https://en.wikipedia.org/wiki/February");
    assertTrue(art.isPresent());
    assertEquals("Q109", art.get().getId());
    art = index.queryWikipediaURL("http://en.wikipedia.org/wiki/February"); // with http
    assertTrue(art.isPresent());
    assertEquals("Q109", art.get().getId());
    art = index.queryWikipediaURL("en.wikipedia.org/wiki/February"); // no http
    assertTrue(art.isPresent());
    assertEquals("Q109", art.get().getId());
    art = index.queryWikipediaURL("https://en.wikipedia.org/wiki/February#Month-long_observances"); // with anchor
    assertTrue(art.isPresent());
    assertEquals("Q109", art.get().getId());
    art = index.queryWikipediaURL("https://en.wikipedia.org/wiki/United_Kingdom"); // with space
    assertTrue(art.isPresent());
    assertEquals("Q145", art.get().getId());
    art = index.queryWikipediaURL("https://en.wikipedia.org/wiki/United Kingdom"); // with space
    assertTrue(art.isPresent());
    assertEquals("Q145", art.get().getId());
    art = index.queryWikipediaURL("https://en.wikipedia.org/wiki/United%20Kingdom"); // escaped
    assertTrue(art.isPresent());
    assertEquals("Q145", art.get().getId());
    art = index.queryWikipediaURL("https://en.wikipedia.org/wiki/Gallo-Roman_Museum_of_Lyon-Fourvière"); // with unicode
    assertTrue(art.isPresent()); 
    assertEquals("Q509", art.get().getId());
    art = index.queryWikipediaURL("https://en.wikipedia.org/wiki/Gallo-Roman_Museum_of_Lyon-Fourviere"); // missing unicode in query
    assertFalse(art.isPresent()); 
    
    art = index.queryWikipediaPage("Hôtel_de_Ville,_Lyon");
    assertTrue(art.isPresent()); 
    assertEquals("Q516", art.get().getId());
    
    art = index.queryWikipediaPage("Hôtel de Ville, Lyon");
    assertTrue(art.isPresent()); 
    assertEquals("Q516", art.get().getId());
    
    art = index.queryWikipediaPage("Hôtel de Ville");
    assertFalse(art.isPresent()); 
    
    art = index.queryWikipediaPage("Hotel_de_Ville,_Lyon"); // missing unicode
    assertFalse(art.isPresent()); 
  }
  
  @Test
  public void testGetAll() throws IOException {
    Resource file = Resource.fromJAR("models/Articles_en_Wikidata_250_20170828.txt");
    LuceneArticleIndex index = ArticleIndexFactory.loadWikiDataIndex(file);

    Collection<String> list = index.getAllArticleIDs();
    assertTrue(list.containsAll(Arrays.asList("Q64", "Q145", "Q276", "Q306", "Q381", "Q516")));
    assertFalse(list.contains("Q1000"));
    assertEquals(250, list.size());
    
    list = index.getAllArticleTitles();
    assertTrue(list.containsAll(Arrays.asList("Berlin", "Ubuntu")));
    assertFalse(list.contains("Ubuntu Linux"));
    assertFalse(list.contains("Ubuntu_(operating_system)"));
    assertEquals(250, list.size());
    
    list = index.getAllArticleNames();
    assertTrue(list.containsAll(Arrays.asList("Berlin", "Ubuntu", "Ubuntu Linux", "La Fleche", "La Flèche")));
    assertFalse(list.contains("Ubuntu_(operating_system)"));
    assertTrue(list.size() > 250);
    
    list = index.getAllArticleTerms();
    assertEquals(0, list.size()); // not used in example file
    
    list = index.getAllArticleURLs();
    //assertTrue(list.containsAll(Arrays.asList("Q64", "Q145", "Q276", "Q306", "Q381", "Q516")));
    //assertFalse(list.contains("Q999"));
    assertEquals(250, list.size());
    
  }
  
}
