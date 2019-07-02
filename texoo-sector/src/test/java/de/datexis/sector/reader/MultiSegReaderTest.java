package de.datexis.sector.reader;

import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.sector.model.SectionAnnotation;
import org.junit.Test;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class MultiSegReaderTest {
  
  public MultiSegReaderTest() {
  }
  
  @Test
  public void testDocument0Reader() throws IOException {
    Resource file = Resource.fromJAR("testdata/222.0");
    Document doc = new MultiSegReader()
        .readDocumentFromFile(file);
    
    assertEquals("222.0", doc.getId());
    List<SectionAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class).sorted().collect(Collectors.toList());
    for(int i = 0; i < 11; i++) {
      assertEquals(Integer.toString(i), anns.get(i).getSectionHeading());
      assertEquals(doc.getSentence(i).getBegin(), anns.get(i).getBegin());
      assertEquals(doc.getSentence(i).getEnd(), anns.get(i).getEnd());
    }
    assertEquals("I finally decided to try alternative medicine.", doc.getSentenceAtPosition(anns.get(2).getBegin()).get().getText());
    assertEquals("I'm a little nervous about trying these unorthodox treatments, but after hearing about Amelia's and Ray's experiences, I'm willing to give them a try. Nothing has worked so far, so what do I have to lose?", doc.getSentence(10).getText());
    assertEquals(11, doc.countSentences());
    assertEquals(11, doc.countAnnotations());

  }
  
  @Test
  public void testDocument1Reader() throws IOException {
    Resource file = Resource.fromJAR("testdata/222.1");
    Document doc = new MultiSegReader()
        .readDocumentFromFile(file);
    
    assertEquals("222.1", doc.getId());
    List<SectionAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class).sorted().collect(Collectors.toList());
    // first section is not annotated, so we skip it!
    /*assertEquals("0", anns.get(0).getSectionHeading());
    assertEquals(doc.getSentence(0).getBegin(), anns.get(0).getBegin());
    assertEquals(doc.getSentence(4).getEnd(), anns.get(0).getEnd());*/
    assertEquals("0", anns.get(0).getSectionHeading());
    assertEquals(doc.getSentence(5 - 5).getBegin(), anns.get(0).getBegin());
    assertEquals(doc.getSentence(5 - 5).getEnd(), anns.get(0).getEnd());
    assertEquals("1", anns.get(1).getSectionHeading());
    assertEquals(doc.getSentence(6 - 5).getBegin(), anns.get(1).getBegin());
    assertEquals(doc.getSentence(16 - 5).getEnd(), anns.get(1).getEnd());
    assertEquals("2", anns.get(2).getSectionHeading());
    assertEquals(doc.getSentence(17 - 5).getBegin(), anns.get(2).getBegin());
    assertEquals(doc.getSentence(17 - 5).getEnd(), anns.get(2).getEnd());
    assertEquals("3", anns.get(3).getSectionHeading());
    assertEquals(doc.getSentence(18 - 5).getBegin(), anns.get(3).getBegin());
    assertEquals(doc.getSentence(28 - 5).getEnd(), anns.get(3).getEnd());
    //assertNotEquals(11, doc.countAnnotations()); // there is an extra 0 annotation at the beginning
    assertEquals(11, doc.countAnnotations());
    assertEquals(81 - 6, doc.countSentences());

  }
  
}
