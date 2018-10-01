package de.datexis.examples;

import de.datexis.ner.MentionAnnotator;
import de.datexis.common.ObjectSerializer;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.ner.GenericMentionAnnotator;
import de.datexis.ner.MentionAnnotation;
import de.datexis.preprocess.DocumentFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Example: how to annotate text using the packaged NER model.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AnnotateEntityRecognition {

  protected final static Logger log = LoggerFactory.getLogger(AnnotateEntityRecognition.class);

  public static void main(String[] args) {
    
    // Parse text into a Document
    Document doc = DocumentFactory.fromText("TeXoo ist ein Framework der Gruppe DATEXIS an der Beuth Hochschule für Technik Berlin.");
    
    // Retrieve the first Document and print
    System.out.println(String.format("Document [%s]: \"%s\"", doc.getLanguage(), doc.getText()));
    
    // Load an Annotator model
    MentionAnnotator annotator = GenericMentionAnnotator.create();
    
    // Annotate Document using the pre-trained model
    annotator.annotate(doc);
    
    // Retrieve all Annotations and print
    doc.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
        System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
        System.out.println(ObjectSerializer.getJSON(ann));
    });
    
    
  }
  
}
