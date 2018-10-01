package de.datexis.examples;

import de.datexis.common.DialogHelpers;
import de.datexis.ner.MentionAnnotator;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.index.ArticleIndexFactory;
import de.datexis.index.impl.VectorArticleIndex;
import de.datexis.nel.NamedEntityAnnotation;
import de.datexis.nel.NamedEntityAnnotator;
import de.datexis.ner.GenericMentionAnnotator;
import de.datexis.preprocess.DocumentFactory;
import java.io.IOException;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Example: how to annotate text using the packaged Entity Linking model.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AnnotateEntityLinking {

  protected final static Logger log = LoggerFactory.getLogger(AnnotateEntityLinking.class);

  private final Resource modelPath = Resource.fromConfig("de.datexis.path.models");
  
  /**
   * This example requires a Knowledge Base and ParVec model stored on disk.
   * @param args : [0] full path to Knowledge Base, e.g. /home/texoo/Articles_de_Wikidata_20170828.json.gz
   *                [1] full path to ParVec Model, e.g. /home/texoo/ParVec_en_100_Wikidata+minlc_20171031.zip
   */
  public static void main(String[] args) throws IOException {
    String modelDir = args.length > 0 ? args[0] : 
        DialogHelpers.askForFile("Please choose a Knowledge Base", new FileNameExtensionFilter("Knowledge Base", "json.gz", "json"));
    String datasetFile = args.length > 1 ? args[1] : 
        DialogHelpers.askForFile("Please choose a ParVec Model", new FileNameExtensionFilter("Paragraph Vectors archive", "zip"));
    new AnnotateEntityLinking().annotateEntities(modelDir, datasetFile);
  }
  
  public void annotateEntities(String knowledgeBaseFile, String parVecFile) throws IOException {
    
    // Parse text into a Document
    Document doc = DocumentFactory.fromText("TeXoo ist ein Framework der Gruppe DATEXIS an der Beuth Hochschule für Technik Berlin.");
    
    // Retrieve the first Document and print
    System.out.println(String.format("Document [%s]: \"%s\"", doc.getLanguage(), doc.getText()));
    
    // Load knowledge base and entity vectors, will build an index in cache directory
    MentionAnnotator ner = GenericMentionAnnotator.create();
    VectorArticleIndex search = ArticleIndexFactory.loadWikiDataIndex(
        Resource.fromFile(knowledgeBaseFile), 
        Resource.fromFile(parVecFile), 
        modelPath.resolve("cache")
    );
    NamedEntityAnnotator annotator = new NamedEntityAnnotator(ner, search);
    
    // Annotate Document using the pre-trained model
    annotator.annotate(doc);
    
    // Retrieve all Annotations and print
    doc.streamAnnotations(Annotation.Source.PRED, NamedEntityAnnotation.class).forEach(ann -> {
        System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
        System.out.println(ObjectSerializer.getJSON(ann));
    });
    
    
  }
  
}
