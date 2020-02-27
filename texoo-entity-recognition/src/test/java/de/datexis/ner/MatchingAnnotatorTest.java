package de.datexis.ner;

import com.google.common.collect.Lists;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MatchingAnnotatorTest {
  
  final String text = "Cancer immunotherapy (sometimes called immuno-oncology) is the use of the immune system to treat cancer. "
      + "Immunotherapies can be categorized as active, passive or hybrid (active and passive). "
      + "These approaches exploit the fact that cancer cells often have molecules on their surface that can be detected by "
      + "the immune system, known as tumour-associated antigens (TAAs); they are often proteins or other macromolecules (e.g. carbohydrates). "
      + "Active immunotherapy directs the immune system to attack tumor cells by targeting TAAs. Passive immunotherapies enhance existing "
      + "anti-tumor responses and include the use of monoclonal antibodies, lymphocytes and cytokines.";
  
  final String[] list = { "tum", "cancer", "immune system", "molecules", "TAA", "immunotherapy", "tumor", "tumor cells", "e" } ;
  
  final String[] uppercase = { "NOT", "TEST" } ;
  final String[] lowercase = { "not", "test" } ;
  
  MatchingAnnotator ann;
  
  
  @Test
  public void testCaseSensitive() {
    
    Document doc = DocumentFactory.fromText(text);
    assertEquals(0, doc.countAnnotations());
    
    ann = new MatchingAnnotator();
    ann.annotate(doc);
    assertEquals(0, doc.countAnnotations());
    
    ann.loadTermsToMatch(Lists.newArrayList(list));
    assertEquals(9, ann.countTerms());
    
    ann.annotate(doc);
    
    doc.streamAnnotations(Annotation.Source.SILVER, MentionAnnotation.class).forEach(ann -> {
      System.out.println(ann.getText());
    });
    
    assertEquals(0, doc.countAnnotations(Annotation.Source.GOLD));
    assertEquals(0, doc.countAnnotations(Annotation.Source.PRED));
    assertEquals(8, doc.countAnnotations(Annotation.Source.SILVER));
    
    
    /*
    cancer 2 
    immune system 3
    molecules 1
    TAA 0 / 0 -> TAAs 2
    immunotherapy 2 -> immunotherapies 1
    e 0
    tumor cells 1
    tumor 0
    */
    
  }
  
  @Test
  public void testLowercase() {
    
    Document doc = DocumentFactory.fromText(text);
    assertEquals(0, doc.countAnnotations());
    
    ann = new MatchingAnnotator(MatchingAnnotator.MatchingStrategy.LOWERCASE);
    ann.annotate(doc);
    assertEquals(0, doc.countAnnotations());
    
    ann.loadTermsToMatch(Lists.newArrayList(list));
    assertEquals(8, ann.countTerms()); // "e" is too short
    
    ann.annotate(doc);
    
    doc.streamAnnotations(Annotation.Source.SILVER, MentionAnnotation.class).forEach(ann -> {
      System.out.println(ann.getText());
    });
    
    assertEquals(0, doc.countAnnotations(Annotation.Source.GOLD));
    assertEquals(0, doc.countAnnotations(Annotation.Source.PRED));
    assertEquals(10, doc.countAnnotations(Annotation.Source.SILVER));
    
    /*
    cancer 3
    immune system 3
    molecules 1
    TAA 0 -> TAAs  2
    immunotherapy 1 -> immunotherapies 2
    e 0
    tumor cells 1
    tumor 0
    */
    
    ann = new MatchingAnnotator(MatchingAnnotator.MatchingStrategy.LOWERCASE);
    ann.loadTermsToMatch(Lists.newArrayList(uppercase));
    doc = DocumentFactory.fromText("This is not a test. Really?");
    Document doc2 = DocumentFactory.fromText("This is NOT a TEST");
    ann.annotate(doc);
    ann.annotate(doc2);
    assertEquals(0, doc.countAnnotations());
    assertEquals(2, doc2.countAnnotations());
    ann.loadTermsToMatch(Lists.newArrayList(lowercase));
    ann.annotate(doc);
    assertEquals(2, doc.countAnnotations());
    
  }
  
  @Test
  public void testLowercaseMethod() {
    String text = "The Immune System, I have known as Tumour-Associated Antigens (TAA). Antigens are often Proteins or other Macromolecules (e.g. carbohydrates C544).";
    String lctx = "The immune system, I have known as tumour-associated antigens (TAA). antigens are often proteins or other macromolecules (e.g. carbohydrates C544).";
    MatchingAnnotator ann = new MatchingAnnotator(MatchingAnnotator.MatchingStrategy.LOWERCASE);
    assertEquals(lctx, ann.convertToLowercase(text));
  }
  
  @Test
  public void testCars() {
    String text = "Der Volkswagen Caddy (2K), auch VW Caddy Life genannt, ist ein PKW-Modell der Marke Volkswagen Nutzfahrzeuge.";
    final String[] list = { "Volkswagen Caddy", "VW", "Caddy Life", "IST" };
    MatchingAnnotator ann = new MatchingAnnotator(MatchingAnnotator.MatchingStrategy.LOWERCASE, Annotation.Source.SILVER, "CAR", 2);
    ann.loadTermsToMatch(Lists.newArrayList(list));
    assertEquals(4, ann.countTerms());
    Document doc = DocumentFactory.fromText(text);
    ann.annotate(doc);
    List<MentionAnnotation> matches = doc.streamAnnotations(Annotation.Source.SILVER, MentionAnnotation.class).sorted().collect(Collectors.toList());
    doc.streamAnnotations(Annotation.Source.SILVER, MentionAnnotation.class).forEach(a -> {
      System.out.println(a.getText());
      assertEquals("CAR", a.getType());
    });
    assertEquals("Volkswagen Caddy", matches.get(0).getText());
    assertEquals("VW", matches.get(1).getText()); // "VW Caddy" would also be ok, but then we're missing "Caddy Life"
    assertEquals("Caddy Life", matches.get(2).getText());
    assertEquals("CAR", matches.get(0).getType());
    assertEquals(3, matches.size());
  }
  
}
