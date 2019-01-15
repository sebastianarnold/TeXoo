package de.datexis.ner;

import com.google.common.collect.Lists;
import de.datexis.annotator.AnnotatorFactory;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sample;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import java.io.IOException;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class GenericMentionAnnotatorTest {
  
  final String textETC = "When available as a diagnostic tool, single photon emission computed tomography (SPECT) and positron emission tomography (PET) neuroimaging may be used to confirm a diagnosis of multi-infarct dementia in conjunction with evaluations involving mental status examination etc. In a person already having dementia, SPECT appears to be superior in differentiating multi-infarct dementia from Alzheimer's disease, compared to the usual mental testing and medical history analysis.";
  final String textEN = "\"Mothers of the Disappeared\" is a song by the rock band U2, the closing track on their album The Joshua Tree, released on 9 March 1987. The song was inspired by lead singer Bono's experiences in Nicaragua and El Salvador in July 1986, following U2's involvement on Amnesty International's A Conspiracy of Hope tour. He learned of the Madres de Plaza de Mayo, a group of women whose children had been abducted with the consent of Argentine and Chilean dictatorships. Thematically the song has been interpreted as an examination of failures and contradictions in US foreign policy and as a criticism of the Reagan Administration, which backed two South American regimes that seized power during coups and which provided financial support for the military regime in El Salvador. The song was favourably received by critics, and has been performed live on several tours, starting with the 1987 Joshua Tree Tour. It was played at four concerts on the 1998 PopMart Tour in South America; during two of these, the Madres joined the band onstage.";
  final String textDE ="Der Nerother Wandervogel ist ein reiner Jungenbund im Geiste des Wandervogelgründers Karl Fischer. Als geistige Grundlage gelten hierbei seine Weistümer – eine Sammlung von Werten und Erkenntnissen. Inhalt des Nerother Wandervogels sind unter anderem Gruppenstunden, Pflege von Volks- und eigenem Liedgut, Laienspiel und das gemeinsame Bauen an der Rheinischen Jugendburg. Besonderes Augenmerk liegt jedoch auf den Wanderfahrten im In- und Ausland. Die gemeinsamen Fahrten stärken die Freundschaft, welche als Wert das Prinzip des NWV als Lebensbund erst möglich macht.";
  final String textFR = "Wikipédia est un projet d’encyclopédie collective établie sur Internet, universelle, multilingue et fonctionnant sur le principe du wiki. Wikipédia a pour objectif d’offrir un contenu librement réutilisable, objectif et vérifiable, que chacun peut modifier et améliorer.";
  
  MentionAnnotator ann;
  
  @Before
  public void init() {
    ann = GenericMentionAnnotator.create();
  }
  
  @Test
  public void testAnnotate() {
    
    Document doc = ann.annotate(textETC);
    // Retrieve the first Document and print
    System.out.println(String.format("Document [%s]: \"%s\"", doc.getLanguage(), doc.getText()));
    // Retrieve all Annotations and print
    doc.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
      System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
      Assert.assertEquals(textETC.substring(ann.getBegin(), ann.getEnd()), ann.getText());
    });
   
    doc = ann.annotate(textEN);
    // Retrieve the first Document and print
    System.out.println(String.format("Document [%s]: \"%s\"", doc.getLanguage(), doc.getText()));
    // Retrieve all Annotations and print
    doc.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
      System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
      Assert.assertEquals(textEN.substring(ann.getBegin(), ann.getEnd()), ann.getText());
    });
    
    doc = ann.annotate(textDE);
    // Retrieve the first Document and print
    System.out.println(String.format("Document [%s]: \"%s\"", doc.getLanguage(), doc.getText()));
    // Retrieve all Annotations and print
    doc.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
      System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
      Assert.assertEquals(textDE.substring(ann.getBegin(), ann.getEnd()), ann.getText());
    });

    doc = ann.annotate(textFR);
    // Retrieve the first Document and print
    System.out.println(String.format("Document [%s]: \"%s\"", doc.getLanguage(), doc.getText()));
    // Retrieve all Annotations and print
    doc.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
      System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
      Assert.assertEquals(textFR.substring(ann.getBegin(), ann.getEnd()), ann.getText());
    });
    
  }
  
  @Test
  public void testAnnotateSample() {
    
    Document doc1 = DocumentFactory.fromText(textEN);
    Document doc2 = DocumentFactory.fromText(textDE);
    
    List<Sentence> sents = Lists.newArrayList(doc1.getSentence(0), doc1.getSentence(1), doc2.getSentence(0), doc2.getSentence(1));

    Sample sample = new Sample(sents, false);
    
    Assert.assertEquals(0, doc1.countAnnotations());
    Assert.assertEquals(0, doc2.countAnnotations());
    Assert.assertEquals(4, sample.countSentences());
    Assert.assertEquals(0, sample.countAnnotations());
    //Assert.assertEquals(4, sample.countTokens());
    
    ann.annotate(sample);
    
    // Retrieve all Annotations and print
    doc1.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
      //System.out.println(String.format("-- doc1 -- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
    });
    doc2.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
      //System.out.println(String.format("-- doc2 -- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
    });
    
    Assert.assertEquals(0, sample.countAnnotations()); // Annotations on Sample should land in their original Documents
    Assert.assertEquals(9, doc1.countAnnotations());
    Assert.assertEquals(7, doc2.countAnnotations());
    
  }
  
  @Test
  public void testAnnotateType() throws IOException {
    
    Resource path = Resource.fromJAR("models");
    MentionAnnotator annotatorEN = (MentionAnnotator) AnnotatorFactory.fromXML(path.resolve("MentionAnnotator_en_NER-GENERIC_WikiNER+tri_20170309")); 
    
    Document doc1 = DocumentFactory.fromText(textEN);
    annotatorEN.getTagger().setType("TEST");
    annotatorEN.annotate(doc1);
    
    // Retrieve all Annotations and print
    doc1.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
      Assert.assertEquals("TEST", ann.getType());
    });
    
  }
  
}
