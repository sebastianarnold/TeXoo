package de.datexis.sector.reader;

import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.sector.model.SectionAnnotation;
import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

public class Wiki727ReaderTest {
  
  public Wiki727ReaderTest() {
  }
  
  @Test
  public void testFullDocument() {
    Resource file = Resource.fromJAR("testdata/Wiki727_2611501.txt");
    Document doc = new Wiki727Reader()
        .withSectionLevel(0)
        .readDocumentFromFile(file);
    assertEquals("Wiki727_2611501.txt", doc.getId());
    List<SectionAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class).sorted().collect(Collectors.toList());
    assertEquals("preface", anns.get(0).getSectionHeading());
    assertNotEquals("History", anns.get(1).getSectionHeading()); // Empty "2. History" should be skipped!
    assertEquals("History | Early history", anns.get(1).getSectionHeading());
    assertEquals(doc.getSentence(12).getBegin(), anns.get(1).getBegin());
    assertEquals(doc.getSentence(16).getEnd(), anns.get(1).getEnd());
    assertEquals("History | Middle Ages", anns.get(2).getSectionHeading());
    assertEquals(doc.getSentence(17).getBegin(), anns.get(2).getBegin());
    assertEquals(doc.getSentence(20).getEnd(), anns.get(2).getEnd());
    assertEquals("Governance", anns.get(6).getSectionHeading());
    assertTrue(doc.getSentenceAtPosition(anns.get(6).getBegin()).get().getText().startsWith("The lowest form of governance"));
    assertEquals(doc.getSentence(54).getBegin(), anns.get(6).getBegin());
    assertEquals(doc.getSentence(62).getEnd(), anns.get(6).getEnd());
    assertEquals(20, doc.countAnnotations());
    assertEquals(163 - 23, doc.countSentences());
  }
  
  @Test
  public void testLevel1() {
    Resource file = Resource.fromJAR("testdata/Wiki727_2611501.txt");
    Document doc = new Wiki727Reader()
        .withSectionLevel(1)
        .readDocumentFromFile(file);
    assertEquals("Wiki727_2611501.txt", doc.getId());
    assertEquals(1, doc.countAnnotations());
    List<SectionAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class).sorted().collect(Collectors.toList());
    assertEquals(doc.getBegin(), anns.get(0).getBegin());
    assertEquals(doc.getEnd(), anns.get(0).getEnd());
  }
  
  @Test
  public void testLevel2() {
    Resource file = Resource.fromJAR("testdata/Wiki727_2611501.txt");
    Document doc = new Wiki727Reader()
        .withSectionLevel(2)
        .readDocumentFromFile(file);
    assertEquals("Wiki727_2611501.txt", doc.getId());
    List<SectionAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class).sorted().collect(Collectors.toList());
    assertEquals("preface", anns.get(0).getSectionHeading());
    assertEquals("History", anns.get(1).getSectionHeading()); // Empty "2. History" should be skipped!
    assertEquals(doc.getSentence(12).getBegin(), anns.get(1).getBegin());
    assertEquals(doc.getSentence(53).getEnd(), anns.get(1).getEnd());
    assertNotEquals("History | Early history", anns.get(2).getSectionHeading());
    assertEquals("Governance", anns.get(2).getSectionHeading());
    assertTrue(doc.getSentenceAtPosition(anns.get(2).getBegin()).get().getText().startsWith("The lowest form of governance"));
    assertEquals(doc.getSentence(54).getBegin(), anns.get(2).getBegin());
    assertEquals(doc.getSentence(62).getEnd(), anns.get(2).getEnd());
    assertEquals(11, doc.countAnnotations());
    assertEquals(163 - 23, doc.countSentences());
  }
  
  @Test
  public void testSkipAbstract() {
    Resource file = Resource.fromJAR("testdata/Wiki727_2611501.txt");
    Document doc = new Wiki727Reader()
        .withSectionLevel(2)
        .withSkipPreface(true)
        .readDocumentFromFile(file);
    assertEquals("Wiki727_2611501.txt", doc.getId());
    List<SectionAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class).sorted().collect(Collectors.toList());
    assertNotEquals("preface", anns.get(0).getSectionHeading());
    assertEquals("History", anns.get(0).getSectionHeading()); // Empty "2. History" should be skipped!
    assertEquals(doc.getSentence(12-12).getBegin(), anns.get(0).getBegin());
    assertEquals(doc.getSentence(53-12).getEnd(), anns.get(0).getEnd());
    assertEquals("Governance", anns.get(1).getSectionHeading());
    assertTrue(doc.getSentenceAtPosition(anns.get(1).getBegin()).get().getText().startsWith("The lowest form of governance"));
    assertEquals(doc.getSentence(54-12).getBegin(), anns.get(1).getBegin());
    assertEquals(doc.getSentence(62-12).getEnd(), anns.get(1).getEnd());
    assertEquals(10, doc.countAnnotations());
    assertNotEquals(163 - 23, doc.countSentences());
    assertEquals(163 - 23 - 12, doc.countSentences());
  }
  
  @Test
  public void testLevel3() {
    Resource file = Resource.fromJAR("testdata/Wiki727_2611501.txt");
    Document doc = new Wiki727Reader()
        .withSectionLevel(3)
        .readDocumentFromFile(file);
    assertEquals("Wiki727_2611501.txt", doc.getId());
    List<SectionAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class).sorted().collect(Collectors.toList());
    assertNotEquals("History", anns.get(1).getSectionHeading()); // Empty "2. History" should be skipped!
    assertEquals("History | Early history", anns.get(1).getSectionHeading());
    assertEquals(doc.getSentence(12).getBegin(), anns.get(1).getBegin());
    assertEquals(doc.getSentence(16).getEnd(), anns.get(1).getEnd());
    assertEquals("History | Middle Ages", anns.get(2).getSectionHeading());
    assertEquals(doc.getSentence(17).getBegin(), anns.get(2).getBegin());
    assertEquals(doc.getSentence(20).getEnd(), anns.get(2).getEnd());
    assertEquals("Governance", anns.get(6).getSectionHeading());
    assertTrue(doc.getSentenceAtPosition(anns.get(6).getBegin()).get().getText().startsWith("The lowest form of governance"));
    assertEquals(doc.getSentence(54).getBegin(), anns.get(6).getBegin());
    assertEquals(doc.getSentence(62).getEnd(), anns.get(6).getEnd());
    assertEquals(20, doc.countAnnotations());
    assertEquals(163 - 23, doc.countSentences());
  }
  
}
