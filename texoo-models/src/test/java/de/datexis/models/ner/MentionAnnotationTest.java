package de.datexis.models.ner;

import com.fasterxml.jackson.databind.ObjectMapper;
import de.datexis.common.ObjectSerializer;

import de.datexis.model.Annotation;
import de.datexis.model.Annotation.Source;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.model.tag.BIOESTag;
import de.datexis.preprocess.DocumentFactory;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Before;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MentionAnnotationTest {
  
  private Dataset data;
  private Document doc;
  
  @Before
  public void init() {
    ArrayList<Token> tokens = new ArrayList<>();
    tokens.add(new Token("Zairean").putTag(Annotation.Source.GOLD, BIO2Tag.B()));
    tokens.add(new Token("Prime").putTag(Annotation.Source.GOLD, BIO2Tag.B()));
    tokens.add(new Token("Minister").putTag(Annotation.Source.GOLD, BIO2Tag.I()));
    tokens.add(new Token("Kengo").putTag(Annotation.Source.GOLD, BIO2Tag.B()));
    tokens.add(new Token("wa").putTag(Annotation.Source.GOLD, BIO2Tag.I()));
    tokens.add(new Token("Dondo").putTag(Annotation.Source.GOLD, BIO2Tag.I()));
    tokens.add(new Token("said").putTag(Annotation.Source.GOLD, BIO2Tag.O()));
    tokens.add(new Token("at").putTag(Annotation.Source.GOLD, BIO2Tag.O()));
    tokens.add(new Token("the").putTag(Annotation.Source.GOLD, BIO2Tag.O()));
    tokens.add(new Token("end").putTag(Annotation.Source.GOLD, BIO2Tag.O()));
    tokens.add(new Token("of").putTag(Annotation.Source.GOLD, BIO2Tag.O()));
    tokens.add(new Token("a").putTag(Annotation.Source.GOLD, BIO2Tag.O()));
    tokens.add(new Token("visit").putTag(Annotation.Source.GOLD, BIO2Tag.O()));
    tokens.add(new Token(".").putTag(Annotation.Source.GOLD, BIO2Tag.O()));
    doc = DocumentFactory.fromTokens(tokens);
    data = new Dataset("Test");
    data.addDocument(doc);
  }
  
  @Test
  public void testSpanTagConversion() {
    Sentence s = doc.getSentence(0);
    assertEquals(BIO2Tag.B(), s.getToken(0).getTag(Annotation.Source.GOLD, BIO2Tag.class));
    assertEquals(BIOESTag.O(), s.getToken(0).getTag(Annotation.Source.PRED, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(0).getTag(Annotation.Source.GOLD, BIOESTag.class));
    // convert to BIOES
    BIO2Tag.convertToBIOES(data, Annotation.Source.GOLD);
    assertEquals(BIO2Tag.B(), s.getToken(0).getTag(Annotation.Source.GOLD, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(0).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIOESTag.O(), s.getToken(0).getTag(Annotation.Source.PRED, BIOESTag.class));
    assertEquals(BIOESTag.S(), s.getToken(0).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.B(), s.getToken(1).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.E(), s.getToken(2).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.B(), s.getToken(3).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.I(), s.getToken(4).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.E(), s.getToken(5).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(6).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(7).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(8).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(9).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(10).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(11).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(12).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), s.getToken(13).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertTrue(BIOESTag.isCorrect(Annotation.Source.GOLD, s.getTokens()));
    assertTrue(BIOESTag.isCorrect(Annotation.Source.PRED, s.getTokens()));
    // copy to PRED
    for(Token t :  s.getTokens()) {
      t.putTag(Annotation.Source.PRED, t.getTag(Annotation.Source.GOLD, BIOESTag.class));
    }
    assertEquals(BIO2Tag.O(), s.getToken(0).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.B(), s.getToken(0).getTag(Annotation.Source.GOLD, BIO2Tag.class));
    assertEquals(BIOESTag.S(), s.getToken(0).getTag(Annotation.Source.PRED, BIOESTag.class));
    assertEquals(BIOESTag.S(), s.getToken(0).getTag(Annotation.Source.GOLD, BIOESTag.class));
    assertTrue(BIOESTag.isCorrect(Annotation.Source.PRED, s.getTokens()));
    // convert back to BIO2
    BIOESTag.convertToBIO2(data, Annotation.Source.PRED);
    s.getToken(0).putTag(Annotation.Source.PRED, BIOESTag.I());
    assertEquals(BIOESTag.I(), s.getToken(0).getTag(Annotation.Source.PRED, BIOESTag.class));
    assertFalse(BIOESTag.isCorrect(Annotation.Source.PRED, s.getTokens()));
    assertEquals(BIO2Tag.B(), s.getToken(0).getTag(Annotation.Source.PRED, BIO2Tag.class)); // I label needs to be overwritten
    assertEquals(BIO2Tag.B(), s.getToken(1).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.I(), s.getToken(2).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.B(), s.getToken(3).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.I(), s.getToken(4).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.I(), s.getToken(5).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(6).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(7).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(8).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(9).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(10).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(11).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(12).getTag(Annotation.Source.PRED, BIO2Tag.class));
    assertEquals(BIO2Tag.O(), s.getToken(13).getTag(Annotation.Source.PRED, BIO2Tag.class));
  }
  
  public void testEncoderSet() {
    // TODO: Test Encoders
  }
  
  @Test
  public void testAnnotations() {
    assertEquals(0, doc.countAnnotations(Annotation.Source.GOLD, MentionAnnotation.class));
    assertEquals(0, doc.countAnnotations(Annotation.Source.PRED, MentionAnnotation.class));
    assertEquals(0, doc.countAnnotations(Annotation.Source.USER, MentionAnnotation.class));
    ArrayList<Token> tokens = new ArrayList<>();
    Token prime = doc.getSentence(0).getToken(1);
    Token minister = doc.getSentence(0).getToken(2);
    tokens.add(prime);
    tokens.add(minister);
    MentionAnnotation test = new MentionAnnotation(Source.GOLD, tokens);
    assertEquals(prime.getBegin(), test.getBegin());
    assertEquals(minister.getEnd(), test.getEnd());
    assertEquals(14, test.getLength());
    assertEquals("GENERIC", test.getType());
    test.setType("POSITION");
    assertEquals("POSITION", test.getType());
    //assertEquals(2, test.countTokens());
    assertNull(test.getRefId());
    
    doc.addAnnotation(test);
    assertEquals(0, doc.countAnnotations(Annotation.Source.PRED, MentionAnnotation.class));
    assertEquals(0, doc.countAnnotations(Annotation.Source.USER, MentionAnnotation.class));
    assertEquals(0, doc.countAnnotations(Annotation.Source.GOLD, Annotation.class));
    assertEquals(1, doc.countAnnotations(Annotation.Source.GOLD, MentionAnnotation.class));
    Iterator<MentionAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, MentionAnnotation.class).iterator();
    MentionAnnotation ann = anns.next();
    assertEquals(Source.GOLD, ann.getSource());
    assertEquals(test, ann);
    
    // TODO:
    //doc.getVector(encoders)
    //doc.addSentence(s);
    //doc.clearVectors();
  }
  
  @Test
  public void testAnnotationRangeQuery() {
    ArrayList<Token> primeMinisterList = new ArrayList<>();
    ArrayList<Token> kengoList = new ArrayList<>();
    Token prime = doc.getSentence(0).getToken(1);
    Token minister = doc.getSentence(0).getToken(2);
    Token kengo = doc.getSentence(0).getToken(3);
    primeMinisterList.add(prime);
    primeMinisterList.add(minister);
    kengoList.add(kengo);
    MentionAnnotation primeMinisterAnn = new MentionAnnotation(Source.GOLD, primeMinisterList);
    doc.addAnnotation(primeMinisterAnn);
    assertEquals("Prime Minister", primeMinisterAnn.getText());
    MentionAnnotation kengoAnn = new MentionAnnotation(Source.GOLD, kengoList);
    doc.addAnnotation(kengoAnn);
    assertEquals("Kengo", kengo.getText());
    assertEquals(2, doc.countAnnotations(Annotation.Source.GOLD, MentionAnnotation.class));
    Iterator<MentionAnnotation> anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, 0, 0, true).iterator();
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, 0, 0, false).iterator();
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, prime.getBegin(), prime.getBegin(), true).iterator();
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, prime.getBegin(), prime.getBegin(), false).iterator();
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, prime.getBegin(), prime.getEnd(), true).iterator();
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, prime.getBegin(), prime.getEnd(), false).iterator();
    assertTrue(anns.hasNext());
    assertEquals(anns.next(), primeMinisterAnn);
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, prime.getBegin(), kengo.getBegin(), true).iterator();
    assertTrue(anns.hasNext());
    assertEquals(anns.next(), primeMinisterAnn);
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, prime.getBegin(), kengo.getBegin(), false).iterator();
    assertTrue(anns.hasNext());
    assertEquals(anns.next(), primeMinisterAnn);
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, prime.getBegin(), kengo.getEnd(), true).iterator();
    assertTrue(anns.hasNext());
    assertEquals(anns.next(), primeMinisterAnn);
    assertTrue(anns.hasNext());
    assertEquals(anns.next(), kengoAnn);
    assertFalse(anns.hasNext());
    anns = doc.streamAnnotationsInRange(Source.GOLD, MentionAnnotation.class, prime.getBegin(), kengo.getEnd(), false).iterator();
    assertTrue(anns.hasNext());
    assertEquals(anns.next(), primeMinisterAnn);
    assertTrue(anns.hasNext());
    assertEquals(anns.next(), kengoAnn);
    assertFalse(anns.hasNext());
  }
  
  @Test
  public void testAnnotationSerialization() {
    ArrayList<Token> tokens = new ArrayList<>();
    Token prime = doc.getSentence(0).getToken(1);
    Token minister = doc.getSentence(0).getToken(2);
    tokens.add(prime);
    tokens.add(minister);
    MentionAnnotation ann = new MentionAnnotation(Source.GOLD, prime.getBegin(), minister.getEnd(), "POSITION", tokens);
    ann.setRefId("142");
    try {
      String json = ObjectSerializer.getJSON(ann);
      System.out.println(ann.getClass());
      System.out.println(json);
      Annotation test = ObjectSerializer.readFromJSON(json, Annotation.class);
      System.out.println(test.getClass());
      assertEquals(ann.getClass(), test.getClass());
      assertEquals(ann.getBegin(), test.getBegin());
      assertEquals(ann.getEnd(), test.getEnd());
      assertEquals(ann.getLength(), test.getLength());
      assertEquals(ann.getText(), test.getText());
      assertEquals(ann.getSource(), test.getSource());
      MentionAnnotation test2 = (MentionAnnotation) test;
      assertEquals(ann.getType(), test2.getType());
      assertEquals(ann.getRefId(), test2.getRefId());
      assertEquals(ann, test);
      assertEquals(ann, test2);
    } catch (IOException ex) {
      ex.printStackTrace();
      fail();
    }
  }
  
  @Test
  public void testAnnotationMatching() {
    Sentence s = doc.getSentence(0);
    ArrayList<Token> tkPrimeMinister = new ArrayList<>();
    tkPrimeMinister.add(s.getToken(1));
    tkPrimeMinister.add(s.getToken(2));
    MentionAnnotation annPrimeMinister = new MentionAnnotation(Source.GOLD, tkPrimeMinister);
    MentionAnnotation annPrimeMinister2 = new MentionAnnotation(Source.USER, tkPrimeMinister);
    doc.addAnnotation(annPrimeMinister);
    doc.addAnnotation(annPrimeMinister);
    doc.addAnnotation(annPrimeMinister2);
    annPrimeMinister2.setRefId("25");
    annPrimeMinister2.setType("POSITION");
    ArrayList<Token> tkKengoWaDondo = new ArrayList<>();
    tkKengoWaDondo.add(s.getToken(3));
    tkKengoWaDondo.add(s.getToken(4));
    tkKengoWaDondo.add(s.getToken(5));
    MentionAnnotation annKengoWaDondo = new MentionAnnotation(Source.GOLD, tkKengoWaDondo);
    doc.addAnnotation(annKengoWaDondo);
    ArrayList<Token> tkPrimeMinisterKengo = new ArrayList<>();
    tkPrimeMinisterKengo.add(s.getToken(1));
    tkPrimeMinisterKengo.add(s.getToken(2));
    tkPrimeMinisterKengo.add(s.getToken(3));
    MentionAnnotation annPrimeMinisterKengo = new MentionAnnotation(Source.PRED, tkPrimeMinisterKengo);
    doc.addAnnotation(annPrimeMinisterKengo);
    ArrayList<Token> tkZaireanPrimeMinisterKengoWaDondo = new ArrayList<>();
    tkZaireanPrimeMinisterKengoWaDondo.add(s.getToken(0));
    tkZaireanPrimeMinisterKengoWaDondo.add(s.getToken(1));
    tkZaireanPrimeMinisterKengoWaDondo.add(s.getToken(2));
    tkZaireanPrimeMinisterKengoWaDondo.add(s.getToken(3));
    tkZaireanPrimeMinisterKengoWaDondo.add(s.getToken(4));
    tkZaireanPrimeMinisterKengoWaDondo.add(s.getToken(5));
    MentionAnnotation annZaireanPrimeMinisterKengoWaDondo = new MentionAnnotation(Source.USER, tkZaireanPrimeMinisterKengoWaDondo);
    doc.addAnnotation(annZaireanPrimeMinisterKengoWaDondo);
    ArrayList<Token> tkPrimeMinist = new ArrayList<>();
    tkPrimeMinist.add(s.getToken(1));
    tkPrimeMinist.add(s.getToken(2));
    MentionAnnotation annPrimeMinist = new MentionAnnotation(Source.USER, "Prime Minist", s.getToken(1).getBegin(), s.getToken(2).getEnd() - 2);
    doc.addAnnotation(annPrimeMinist);
    //test.matches(test)
    //test.intersects(test)
    assertTrue(annPrimeMinister.equals(annPrimeMinister));
    assertTrue(annPrimeMinister.matches(annPrimeMinister));
    assertTrue(annPrimeMinister.contains(annPrimeMinister));
    assertTrue(annPrimeMinister.intersects(annPrimeMinister));
    assertFalse(annPrimeMinister.equals(annPrimeMinister2));
    assertTrue(annPrimeMinister.matches(annPrimeMinister2));
    assertTrue(annPrimeMinister.contains(annPrimeMinister2));
    assertTrue(annPrimeMinister.intersects(annPrimeMinister2));
    assertFalse(annPrimeMinister.equals(annKengoWaDondo));
    assertFalse(annPrimeMinister.matches(annKengoWaDondo));
    assertFalse(annPrimeMinister.contains(annKengoWaDondo));
    assertFalse(annPrimeMinister.intersects(annKengoWaDondo));
    assertFalse(annPrimeMinister.equals(annPrimeMinisterKengo));
    assertFalse(annPrimeMinister.matches(annPrimeMinisterKengo));
    assertFalse(annPrimeMinister.contains(annPrimeMinisterKengo));
    assertTrue(annPrimeMinister.intersects(annPrimeMinisterKengo));
    assertFalse(annPrimeMinister.equals(annZaireanPrimeMinisterKengoWaDondo));
    assertFalse(annPrimeMinister.matches(annZaireanPrimeMinisterKengoWaDondo));
    assertFalse(annPrimeMinister.contains(annZaireanPrimeMinisterKengoWaDondo));
    assertTrue(annPrimeMinister.intersects(annZaireanPrimeMinisterKengoWaDondo));
    assertFalse(annPrimeMinister.equals(annPrimeMinist));
    assertFalse(annPrimeMinister.matches(annPrimeMinist));
    assertTrue(annPrimeMinister.contains(annPrimeMinist));
    assertTrue(annPrimeMinister.intersects(annPrimeMinist));
    assertTrue(annZaireanPrimeMinisterKengoWaDondo.contains(annPrimeMinister));
    assertTrue(annZaireanPrimeMinisterKengoWaDondo.contains(annPrimeMinister2));
    assertTrue(annZaireanPrimeMinisterKengoWaDondo.contains(annPrimeMinist));
    
    //doc.countAnnotations(Annotation.Source.GOLD, type)
    //doc.getAnnotations(begin, end)
    
    assertEquals(0, doc.countAnnotations(Source.GOLD, Annotation.class));
    assertEquals(3, doc.countAnnotations(Source.GOLD, MentionAnnotation.class));
    assertEquals(1, doc.countAnnotations(Source.PRED, MentionAnnotation.class));
    assertEquals(3, doc.countAnnotations(Source.USER, MentionAnnotation.class));
    
    
    //Iterables.contains(
  }
  
  @Test
  public void testTagConversion() {
    MentionAnnotation.annotateFromTags(doc, Source.GOLD, BIO2Tag.class);
    assertEquals(3, doc.countAnnotations(Source.GOLD, MentionAnnotation.class));
    assertEquals(BIOESTag.O(), doc.getToken(0).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), doc.getToken(1).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), doc.getToken(2).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), doc.getToken(3).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), doc.getToken(4).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), doc.getToken(5).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), doc.getToken(6).get().getTag(Source.GOLD, BIOESTag.class));
    MentionAnnotation.createTagsFromAnnotations(doc, Source.GOLD, BIOESTag.class);
    assertEquals(BIOESTag.S(), doc.getToken(0).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.B(), doc.getToken(1).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.E(), doc.getToken(2).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.B(), doc.getToken(3).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.I(), doc.getToken(4).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.E(), doc.getToken(5).get().getTag(Source.GOLD, BIOESTag.class));
    assertEquals(BIOESTag.O(), doc.getToken(6).get().getTag(Source.GOLD, BIOESTag.class));
  }
  
  // TODO: test annotation equals
  // TODO: test annotation sorting in Document
  // TODO: test annotation evaluation / match in Document
  
}
