package de.datexis.ner;

import com.google.common.collect.Lists;
import de.datexis.annotator.Annotator;
import de.datexis.annotator.AnnotatorFactory;
import de.datexis.common.Resource;
import de.datexis.encoder.impl.LetterNGramEncoder;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.ner.tagger.MentionTagger;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MentionAnnotatorTest {
  
  final private String text = "Aspirin has an antiplatelet effect (e.g. preventing heart attacts, strokes and blood clot formation) by stopping the binding together of platelets [1]. "
          + "Aspirin is also known as acetylsalicylic acid.";
  
  private Dataset data;
  
  @Before
  public void init() {
    Document doc = DocumentFactory.fromText(text);
    data = new Dataset();
    data.addDocument(doc);
  }
  
  @Test
  public void testTrigramAnnotator() throws IOException {
    
    Resource path = Resource.createTempDirectory();
    //Resource path = ExportHandler.createExportPath("test_trigram");
    
    LetterNGramEncoder trigram = new LetterNGramEncoder(3);
    trigram.setId("TRI");
    assertFalse(trigram.isModelAvailable());
    assertTrue(trigram.isModelAvailableInChildren());
    
    Annotator annotator = new Annotator(trigram);
    assertTrue(annotator.isModelAvailable()); // EmptyTagger is available
    assertTrue(annotator.isModelAvailableInChildren());
    
    assertEquals(3, trigram.getN());
    // TODO: train all missing models using annotator?
    //annotator.trainModel();
    trigram.trainModel(data.getDocuments());
    trigram.saveModel(path, "trigram");
    assertTrue(trigram.isModelAvailable());
    assertTrue(annotator.isModelAvailable());
    assertTrue(annotator.isModelAvailableInChildren());
    
    assertEquals(2, trigram.encode("an").sumNumber().intValue()); // word has 2 trigrams
    
    annotator.writeTrainLog(path);
    annotator.writeTestLog(path);
    annotator.writeModel(path);
    
    Annotator test = AnnotatorFactory.loadAnnotator(path);
    //assertEquals(Annotator.class, test.getClass());
    assertTrue(annotator.isModelAvailable()); // EmptyTagger is available
    assertTrue(annotator.isModelAvailableInChildren());
    
  }
  
  @Test
  public void testMentionAnnotator() throws IOException {
    
    Resource path = Resource.createTempDirectory();
    
    // build model
    LetterNGramEncoder trigram = new LetterNGramEncoder("TRI")
            .setN(3);
    trigram.trainModel(data.getDocuments());
    trigram.saveModel(path, "trigram");
    
    MentionTagger tagger = new MentionTagger("BLSTM");
    tagger.setTagset(BIO2Tag.class, "TEST");
    tagger.setEncoders(Lists.newArrayList(trigram));
    tagger.setModelParams(50, 20, 1, 0.01);
    tagger.setTrainingParams(5, 1, false);
    tagger.trainModel(data);
    tagger.saveModel(path, "ner");
    
    // create annotator
    MentionAnnotator ann = new MentionAnnotator(tagger);
    ann.addComponent(trigram);
    ann.getProvenance()
        .setLanguage("en")
        .setTask("NER")
        .setDataset("en_Test")
        .setFeatures("tri");
    
    // save model
    ann.writeTrainLog(path);
    ann.writeTestLog(path);
    ann.writeModel(path, "ner_test");
    
    assertEquals("MentionAnnotator_NER@en_Test+tri_" + ann.getProvenance().getDate(), ann.getProvenance().toString());
    
    // test model
    assertEquals(MentionAnnotator.class, ann.getClass());
    assertEquals(MentionTagger.class, ann.getTagger().getClass());
    assertTrue(ann.isModelAvailable());
    assertTrue(ann.isModelAvailableInChildren());
    ann.getTagger().tag(data.getDocuments());
    
    // load saved model
    Annotator test = AnnotatorFactory.loadAnnotator(path);
    
    // test saved model
    assertEquals(MentionAnnotator.class, test.getClass());
    assertEquals(MentionTagger.class, test.getTagger().getClass());
    assertTrue(test.isModelAvailable());
    assertTrue(test.isModelAvailableInChildren());
    test.getTagger().tag(data.getDocuments());
    
    assertEquals(ann.getProvenance().toString(), test.getProvenance().toString());
  }
  
}
