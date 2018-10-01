package de.datexis.encoder;

import org.junit.Test;
import de.datexis.encoder.impl.LetterNGramEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.model.tag.BIO2Tag;
import java.util.ArrayList;
import java.util.List;
import static org.junit.Assert.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author sarnold
 */
public class NGramEncoderTest {
  
  private final LetterNGramEncoder ngram;
  private final LetterNGramEncoder sgram;
  
  public NGramEncoderTest() {
    // initialize without reflection to get better error messages
    ngram = new LetterNGramEncoder(3);
    sgram = new LetterNGramEncoder(2);
  }
  
  @Test
  public void testAlphabet() {
    
    String text = "Test";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "(ASA)";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "Reye's";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
            
    text = "acetaminophen/aspirin/pro-caffeine";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "11.8%";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "1,667";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "a b";
    assertEquals("ab", ngram.keepOnlyPrintableChars(text));
    
    text = "25 °C (77 °F)";
    assertEquals("25c(77f)", ngram.keepOnlyPrintableChars(text));
    
    text = "\"<cite>\"";
    assertEquals(text, ngram.keepOnlyPrintableChars(text));
    
    text = "„(Quelle:http://example.com;Datum)“";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "§63";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "Maßähnliche";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "né";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "né";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "français";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "l'amuïssement";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "conquête";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
    text = "?";
    assertEquals(text.toLowerCase(), ngram.keepOnlyPrintableChars(text));
    
  }
  
  @Test
  public void testNGramGeneration() {

    String text = "Aspirin";

    List<String> grams = ngram.generateNGrams(text, 3);
    assertEquals("#as", grams.get(0));
    assertEquals("asp", grams.get(1));
    assertEquals("in#", grams.get(6));
    assertEquals(7, grams.size());

    grams = ngram.generateNGrams(text, 4);
    assertEquals("#asp", grams.get(0));
    assertEquals("rin#", grams.get(5));
    assertEquals(6, grams.size());
    
    text = "cat";
    grams = ngram.generateNGrams(text, 5);
    assertEquals("#cat#", grams.get(0));
    assertEquals(1, grams.size());
    
  }
  
  @Test
  public void testSimpleNGramGeneration() {

    String text = "Aspirin";

    List<String> grams = sgram.generateNGrams(text, 3);
    assertEquals("#as", grams.get(0));
    assertEquals("asp", grams.get(1));
    assertEquals("in#", grams.get(6));
    assertEquals(7, grams.size());

    grams = sgram.generateNGrams(text, 4);
    assertEquals("#asp", grams.get(0));
    assertEquals("rin#", grams.get(5));
    assertEquals(6, grams.size());
    
    // N-grams should retain only first char uppercase
    text = "Cat";
    grams = sgram.generateNGrams(text, 5);
    assertEquals("#cat#", grams.get(0));
    assertEquals(1, grams.size());
    
  }
  
  @Test
  public void testNGramEncoder() {
    
    Dataset data = createTestData();
    ngram.trainModel(data.getDocuments());

    /* 
      #Zaimean#
      #Prime#
      #Minister#
      #Kisto#
      25 3-grams, 24 unique, 2 twice (ime) (ist)
    */
    
    assertEquals(23, ngram.getVectorSize());
    
    assertFalse(ngram.isUnknown("Prime"));
    assertTrue(ngram.isUnknown("Kengo"));
    //assertFalse(ngram.isUnknown("Pristo"));
    
    INDArray vec1 = ngram.encode(new Token("Minister"));
    INDArray vec2 = ngram.encode(new Token("Mistister"));
    System.out.println(vec1.toString());
    System.out.println(vec2.toString());
    assertEquals(23, vec1.length());
    assertEquals(23, vec2.length());
    assertEquals(1.0, vec1.maxNumber().doubleValue(), 0.);
    assertEquals(1.0, vec2.maxNumber().doubleValue(), 0.); // no 2 for "ist"
    assertEquals(8.0, vec1.sumNumber().doubleValue(), 0.);
    assertEquals(5.0, vec2.sumNumber().doubleValue(), 0.);
    
  }
  
  @Test
  public void testEncodings() {
    Dataset data = createTestData();
    ngram.trainModel(data.getDocuments());
    INDArray a = ngram.encode("Minister");
    // this has to pass for all Encoders. Don't change!
    long size = ngram.getVectorSize();
    assertEquals(size, a.length());
    assertEquals(size, a.size(0));
    assertEquals(size, a.rows());
    assertEquals(1, a.size(1));
    assertEquals(1, a.columns());
    assertEquals(2, a.rank());
  }
  
  private Dataset createTestData() {
    
    Dataset data = new Dataset("Test");
    
    ArrayList<Token> tokens = new ArrayList<>();
    tokens.add(new Token("Zaimean").putTag(Annotation.Source.GOLD, new BIO2Tag(BIO2Tag.Label.B)));
    tokens.add(new Token("Prime").putTag(Annotation.Source.GOLD, new BIO2Tag(BIO2Tag.Label.I)));
    tokens.add(new Token("Minister").putTag(Annotation.Source.GOLD, new BIO2Tag(BIO2Tag.Label.I)));
    tokens.add(new Token("Kisto").putTag(Annotation.Source.GOLD, new BIO2Tag(BIO2Tag.Label.O)));
    
    Document doc = new Document();
    Sentence s = new Sentence(tokens);
    doc.addSentence(s);
    data.addDocument(doc);
    
    return data;
    
  }
    
}
