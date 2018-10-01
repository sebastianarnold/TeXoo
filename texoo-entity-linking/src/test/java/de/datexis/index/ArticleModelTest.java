package de.datexis.index;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.datexis.model.Article;
import java.io.IOException;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Before;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ArticleModelTest {
  
  String title = "Database Systems and Text-based Information Systems (DATEXIS)";
  
  String text = "The group \"Database Systems and Text-based Information Systems (DATEXIS)\" at Beuth Hochschule für Technik Berlin "
          + "conducts research and teaching in managing text-based and structured data. We address areas like explorative analytics, "
          + "information extraction, information gathering and information pricing.";
  
  ObjectMapper mapper = new ObjectMapper();
  
  @Before
  public void init() {
    
   
  }
  
  @Test
  public void testArticleModel() {
    Article article = new Article("23", "http://www.datexis.com", title, text);
    assertEquals("23", article.getId());
    assertEquals(title, article.getTitle());
    assertEquals(text, article.getText());
    assertEquals(0, article.getBegin());
    assertEquals(text.length(), article.getLength());
    assertNull(article.getDescription());
    assertNotNull(article.getNames());
    assertTrue(article.getNames().isEmpty());
    assertNotNull(article.getTerms());
    assertTrue(article.getTerms().isEmpty());
    article.addName("Datexis");
    assertFalse(article.getNames().isEmpty());
    assertTrue(article.getNames().contains("Datexis"));
    assertEquals(1, article.getNames().size());
    article.addName("Datexis");
    assertEquals(1, article.getNames().size());
    article.addName("DATEXIS");
    assertEquals(2, article.getNames().size());
    ArticleRef candidate = new ArticleRef(article);
    assertEquals("23", candidate.getId());
    assertEquals(title, candidate.getTitle());
  }
  
  @Test
  public void testArticleSerialization() {
    Article article = new Article("23", "http://www.datexis.com", title, text);
    ArticleRef candidate = new ArticleRef(article);
    try {
      String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(article);
      System.out.println(article.getClass());
      System.out.println(json);
    } catch (IOException ex) {
      fail();
    }
    try {
      String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(candidate);
      System.out.println(candidate.getClass());
      System.out.println(json);
    } catch (IOException ex) {
      fail();
    }
  }
  
}
