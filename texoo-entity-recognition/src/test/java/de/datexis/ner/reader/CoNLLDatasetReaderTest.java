package de.datexis.ner.reader;

import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.ner.MentionAnnotation;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class CoNLLDatasetReaderTest {
  
  public CoNLLDatasetReaderTest() {
    
  }
  
  @Test
  public void testCoNLL2003() throws IOException {
    
    Resource path = Resource.fromJAR("datasets/CoNLL2003.conll");
    
    Dataset data = new CoNLLDatasetReader()
        .withFirstSentenceAsTitle(true)
        .withName("CoNLL2003")
        .read(path);
    
    assertEquals("CoNLL2003", data.getName());
    assertEquals( 2, data.countDocuments());
    assertEquals(13, data.countSentences());
    assertEquals(85, data.countTokens());
    assertEquals(16, data.countAnnotations(Annotation.Source.GOLD));
    
    assertEquals("SOCCER - INTERNATIONAL GAME RESULT.", data.getDocument(0).get().getTitle());
    assertEquals("BASKETBALL - ITALY BEAT FRANCE IN UNDER-21 MATCH.", data.getDocument(1).get().getTitle());
    
    List<MentionAnnotation> anns = data.getDocument(0).get().streamAnnotations(MentionAnnotation.class).sorted().collect(Collectors.toList());
    assertEquals("LONDON", anns.get(0).getText());
    assertEquals("LOC", anns.get(0).getType());
    assertEquals("Green Star", anns.get(1).getText());
    assertEquals("ORG", anns.get(1).getType());
    assertEquals("Germany", anns.get(2).getText());
    assertEquals("LOC", anns.get(2).getType());
    
  }
  
  @Test
  public void testCoNLL2003Plain() throws IOException {
    
    Resource path = Resource.fromJAR("datasets/CoNLL2003.conll");
    
    Dataset data = new CoNLLDatasetReader()
        .withAnnotationSource(Annotation.Source.SILVER)
        .withGenericType("TEST")
        .read(path);
    
    assertEquals("CoNLL2003", data.getName());
    assertEquals( 2, data.countDocuments());
    assertEquals(13, data.countSentences());
    assertEquals(85, data.countTokens());
    assertEquals(0, data.countAnnotations(Annotation.Source.GOLD));
    assertEquals(16, data.countAnnotations(Annotation.Source.SILVER));
    
    assertEquals(null, data.getDocument(0).get().getTitle());
    assertEquals(null, data.getDocument(1).get().getTitle());
    
    List<MentionAnnotation> anns = data.getDocument(0).get().streamAnnotations(MentionAnnotation.class).sorted().collect(Collectors.toList());
    assertEquals("LONDON", anns.get(0).getText());
    assertEquals("TEST", anns.get(0).getType());
    assertEquals("Green Star", anns.get(1).getText());
    assertEquals("TEST", anns.get(1).getType());
    assertEquals("Germany", anns.get(2).getText());
    assertEquals("TEST", anns.get(2).getType());
    
  }
  
  @Test
  public void testWikiGold() throws IOException {
    
    Resource path = Resource.fromJAR("datasets/Wikigold.conll");
    
    Dataset data = new CoNLLDatasetReader()
        .withFirstSentenceAsTitle(false)
        .read(path);
    
    assertEquals("Wikigold", data.getName());
    assertEquals( 3, data.countDocuments());
    assertEquals( 4, data.countSentences());
    assertEquals(60, data.countTokens());
    assertEquals( 9, data.countAnnotations(Annotation.Source.GOLD));
    
    List<MentionAnnotation> anns = data.getDocument(0).get().streamAnnotations(MentionAnnotation.class).sorted().collect(Collectors.toList());
    assertEquals("West Germany national soccer team", anns.get(0).getText());
    assertEquals("ORG", anns.get(0).getType());
    assertEquals("West Germany", anns.get(1).getText());
    assertEquals("LOC", anns.get(1).getType());
    
  }
  
  @Test
  public void testWikiNER() throws IOException {
    
    Resource path = Resource.fromJAR("datasets/WikiNER.conll");
    
    Dataset data = new CoNLLDatasetReader()
        .withFirstSentenceAsTitle(false)
        .withGenericType(MentionAnnotation.Type.GENERIC)
        .read(path);
    
    assertEquals("WikiNER", data.getName());
    assertEquals( 1, data.countDocuments());
    //assertEquals( 1, data.countSentences()); // FIXME: ." detected as two sentences
    assertEquals(32, data.countTokens());
    assertEquals( 1, data.countAnnotations(Annotation.Source.GOLD));
    
    List<MentionAnnotation> anns = data.getDocument(0).get().streamAnnotations(MentionAnnotation.class).sorted().collect(Collectors.toList());
    assertEquals("The Oxford Companion to Philosophy", anns.get(0).getText());
    assertEquals(MentionAnnotation.Type.GENERIC, anns.get(0).getType());
    
  }
  
  @Test
  public void testTwitterNER() throws IOException {
    
    Resource path = Resource.fromJAR("datasets/TwitterNER.conll");
    
    Dataset data = new CoNLLDatasetReader()
        .withFirstSentenceAsTitle(false)
        .withGenericType(MentionAnnotation.Type.GENERIC)
        .read(path);
    
    assertEquals("TwitterNER", data.getName());
    assertEquals( 1, data.countDocuments());
    // assertEquals( 3, data.countSentences()); // Sentece splitter splits Dec. 10
    assertEquals(36, data.countTokens());
    assertEquals( 3, data.countAnnotations(Annotation.Source.GOLD));
    
    List<MentionAnnotation> anns = data.getDocument(0).get().streamAnnotations(MentionAnnotation.class).sorted().collect(Collectors.toList());
    assertEquals("ABCD", anns.get(0).getText());
    assertEquals(MentionAnnotation.Type.GENERIC, anns.get(0).getType());
    assertEquals("Test Award", anns.get(1).getText());
    assertEquals(MentionAnnotation.Type.GENERIC, anns.get(1).getType());
    
  }
}
