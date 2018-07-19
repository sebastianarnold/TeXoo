package de.datexis.models.ner;

import de.datexis.annotator.AnnotatorFactory;
import de.datexis.common.Resource;
import de.datexis.model.Document;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A generic Named Entity Mention Annotator with language detection.
 * Uses pre-trained models for German and English.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class GenericMentionAnnotator extends MentionAnnotator {

  protected final static Logger log = LoggerFactory.getLogger(GenericMentionAnnotator.class);
  
  // Configure model path to use internal JAR resources
  private final static Resource path = Resource.fromJAR("models");
  
  private final MentionAnnotator annotatorEN;
  private final MentionAnnotator annotatorDE;
  
  /**
   * Returns a generic Named Entity Mention Annotator with language detection.
   * It uses pre-trained models for German and English.
   */
  public static MentionAnnotator create() {
    try {
      return new GenericMentionAnnotator();
    } catch (IOException ex) {
      log.error("Could not load packaged models for MentionAnnotator.");
      ex.printStackTrace();
      return null;
    }
  }
  
  /**
   * Please use GenericMentionAnnotator().create()
   */
  protected GenericMentionAnnotator() throws IOException {
    annotatorEN = (MentionAnnotator) AnnotatorFactory.fromXML(path.resolve("MentionAnnotator_en_NER-GENERIC_WikiNER+tri_20170309")); // English Wiki Entities
    annotatorDE = (MentionAnnotator) AnnotatorFactory.fromXML(path.resolve("MentionAnnotator_de_NER-GENERIC_WikiNER+tri_20170309")); // German Wiki Entities
  }
  
  /**
   * Annotates given documents for English and German texts.
   */
  @Override
  public void annotate(Collection<Document> docs) {
    Map<String, List<Document>> groups = docs.stream().collect(Collectors.groupingBy(doc -> Optional.ofNullable(doc.getLanguage()).orElse("unk")));
    for(Map.Entry<String, List<Document>> group : groups.entrySet()) {
      if(group.getKey().equals("de")) {
        annotatorDE.annotate(group.getValue());
      } else if(group.getKey().equals("en")) {
        annotatorEN.annotate(group.getValue());
      } else {
        log.warn("Detected language " + group.getKey() + ", using English annotator.");
        annotatorEN.annotate(group.getValue());
      }
    }
  }
  

}
