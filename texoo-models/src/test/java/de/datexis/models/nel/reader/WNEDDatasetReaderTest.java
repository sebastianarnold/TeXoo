package de.datexis.models.nel.reader;

import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.models.index.ArticleIndexFactory;
import de.datexis.models.index.ArticleRef;
import de.datexis.models.index.impl.LuceneArticleIndex;
import de.datexis.models.nel.NamedEntityAnnotation;
import de.datexis.models.ner.MentionAnnotation;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class WNEDDatasetReaderTest {

  private Dataset data;
  private LuceneArticleIndex index;

  @Before
  public void init() throws IOException {

    Resource file = Resource.fromJAR("models/Articles_en_Wikidata_250_20170828.txt");
    index = ArticleIndexFactory.loadWikiDataIndex(file);
    
    Resource xmlPath = Resource.fromJAR("datasets/WNED_LoremIpsum.xml");
    Resource rawTextPath = Resource.fromJAR("datasets/");

    data = new WNEDDataset.Reader(xmlPath, rawTextPath)
        .withAnnotations(NamedEntityAnnotation.class)
        .withAnnotations(MentionAnnotation.class)
        .read();
    
  }

  @Test
  public void hasAllAnnotations() {
    assertNotNull(data);
    assertEquals(1, data.countDocuments());
    assertEquals(10, data.countAnnotations());
  }

  @Test
  public void annotationsAreOnTheRightPosition() {
    for(Document doc : data.getDocuments()) {
      String txt = doc.getText();
      for(NamedEntityAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, NamedEntityAnnotation.class)) {
        int begin = ann.getBegin();
        int end = ann.getEnd();
        String mention = txt.substring(begin, end);
        assertEquals(ann.getText(), mention);
      }
      for(MentionAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, MentionAnnotation.class)) {
        int begin = ann.getBegin();
        int end = ann.getEnd();
        String mention = txt.substring(begin, end);
        assertEquals(ann.getText(), mention);
      }
    }
  }
  
  @Test
  public void annotationsAreInIndex() {
    List<String> expected = Arrays.asList("Q1375", "Q82", "Q175", "Q102");
    List<String> found = new ArrayList<>();
    for(Document doc : data.getDocuments()) {
      for(NamedEntityAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, NamedEntityAnnotation.class)) {
        Optional<ArticleRef> art = index.queryWikipediaPage(ann.getRefId());
        if(art.isPresent()) {
          System.out.println("Found " + ann.getRefId() + ": " + art.get().getUrl() + " (" + art.get().getId() + ")");
          found.add(art.get().getId());
        } else {
          System.out.println("Did not find " + ann.getRefId()+ "!");
        }
      }
    }
    assertArrayEquals(expected.toArray(), found.toArray());
  }

}
