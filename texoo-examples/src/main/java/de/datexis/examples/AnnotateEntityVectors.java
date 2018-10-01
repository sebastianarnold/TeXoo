package de.datexis.examples;

import de.datexis.common.DialogHelpers;
import de.datexis.ner.MentionAnnotator;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.index.encoder.EntityEncoder;
import de.datexis.ner.GenericMentionAnnotator;
import de.datexis.ner.MentionAnnotation;
import de.datexis.preprocess.DocumentFactory;
import java.io.IOException;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Example: how to annotate text using the packaged Entity Vectors model.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AnnotateEntityVectors {

  protected final static Logger log = LoggerFactory.getLogger(AnnotateEntityVectors.class);

  private final Resource modelPath = Resource.fromConfig("de.datexis.path.models");
  
  /**
   * This example requires a ParVec model stored on disk.
   * @param args : [0] full path to ParVec Model, e.g. /home/texoo/ParVec_en_100_Wikidata+minlc_20171031.zip
   */
  public static void main(String[] args) throws IOException {
    String parVecFile = args.length > 0 ? args[0] : 
        DialogHelpers.askForFile("Please choose a ParVec Model", new FileNameExtensionFilter("Paragraph Vectors archive", "zip"));
    new AnnotateEntityVectors().annotateEntities(parVecFile);
  }
  
  public void annotateEntities(String parVecFile) throws IOException {
    
    // Parse text into a Document
    Document doc = DocumentFactory.fromText("TeXoo ist ein Framework der Gruppe DATEXIS an der Beuth Hochschule für Technik Berlin.");
    
    // Retrieve the first Document and print
    System.out.println(String.format("Document [%s]: \"%s\"", doc.getLanguage(), doc.getText()));
    
    // Load knowledge base and entity vectors, will build an index in cache directory
    MentionAnnotator annotator = GenericMentionAnnotator.create();
    EntityEncoder encoder = new EntityEncoder(Resource.fromFile(parVecFile), EntityEncoder.Strategy.NAME);
    
    // Annotate Document using the pre-trained model
    annotator.annotate(doc);
    encoder.encodeEach(doc, Annotation.Source.PRED, MentionAnnotation.class);
    
    // Retrieve all Annotations and print
    doc.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
        System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
        System.out.println(ObjectSerializer.getJSON(ann));
        System.out.println(ann.getVector(EntityEncoder.class));
    });
    
    
  }
  
}
