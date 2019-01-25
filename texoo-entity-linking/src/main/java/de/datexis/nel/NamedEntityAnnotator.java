package de.datexis.nel;

import com.google.common.collect.Lists;
import de.datexis.annotator.Annotator;
import de.datexis.ner.MentionAnnotator;
import de.datexis.common.Timer;
import de.datexis.encoder.Encoder;
import de.datexis.model.Document;
import de.datexis.ner.MentionAnnotation;
import de.datexis.model.Annotation;
import de.datexis.index.ArticleIndex;
import de.datexis.index.ArticleRef;
import de.datexis.index.impl.VectorArticleIndex;
import de.datexis.preprocess.DocumentFactory;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.NavigableMap;
import java.util.TreeMap;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Annotator that detects and links Named Entities.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class NamedEntityAnnotator extends Annotator {

  protected final static Logger log = LoggerFactory.getLogger(NamedEntityAnnotator.class);
  
  protected final MentionAnnotator ner;
  protected final ArticleIndex index;
  protected final Encoder encoder;
    
  public NamedEntityAnnotator(MentionAnnotator recognize, ArticleIndex search, Encoder disambiguate) {
    this.ner = recognize;
    this.index = search;
    this.encoder = disambiguate;
  }
  
  public NamedEntityAnnotator(MentionAnnotator recognize, ArticleIndex search) {
    this.ner = recognize;
    this.index = search;
    this.encoder = null;
  }
  
  @Override
  public Document annotate(String text) {
    log.trace("Annotating document: " + text);
    // update
    Document doc = DocumentFactory.fromText(text);
    if(doc.countTokens() == 0) return doc;
    annotate(doc);
    return doc;
  }
  
  @Override
  public Document annotate(Document doc) {
    annotate(Lists.newArrayList(doc));
    return doc;
  }

  @Override  
  public void annotate(Collection<Document> docs) {
    Timer timer = new Timer();
    timer.start();
    NavigableMap<Integer,NamedEntityAnnotation> result = new TreeMap<>();
    
    // recognize
    timer.resetSplit();
    ner.annotate(docs);
    timer.setSplit("NER");
    
    // search & disambiguate
    for(Document doc : docs) {
      createSignature(doc);
      disambiguateMentions(doc, Annotation.Source.PRED);
    }
    timer.setSplit("NED");
    
    timer.stop();
    log.debug("Annotated " + docs.size() + " documends [" + timer.get("NER") + " NER, " + timer.get("NED") + " NED, " + timer.get() + " total]");
  }
  
  public ArticleIndex getKnowlegeBase() {
    return this.index;
  }

  /**
   * Attaches MentonAnnotations to the Document
   */
  protected void recognizeMentions(Document doc) {
    ner.annotate(doc);
  }

  /**
   * Attaches Vector to the Document
   */
  protected void createSignature(Document doc) {
    //INDArray sig = encoder.encode(doc);
    //doc.putVector(encoder.getClass(), sig);
  }

  /**
   * Attaches NamedEntityAnnotation to the Document
   */
  public void disambiguateMentions(Document doc, Annotation.Source source) {
    //INDArray context = doc.getVector(encoder.getClass());
    List<MentionAnnotation> anns = doc.streamAnnotations(source, MentionAnnotation.class).collect(Collectors.toList());
    for(MentionAnnotation mention : anns) {
      NamedEntityAnnotation entity = new NamedEntityAnnotation(mention, new ArrayList<>());
      //if(annotationExists(ner, doc.ge)) {
      String entityMention = mention.getText();
      String entityContext = doc.getSentenceAtPosition(mention.getBegin()).get().toTokenizedString();
      List<ArticleRef> candidates;
      if(index instanceof VectorArticleIndex) {
        // get many top candidates from Lucene and rerank unsing vectors
        candidates = ((VectorArticleIndex)index).querySimilarArticles(entityMention, entityContext, 1);
      } else {
        // get only top candidate from Lucene
        candidates = index.queryNames(entityMention, 1);
      }
      if(candidates.size() > 0) {
        // TODO: this piece of code should be part of NamedEntityAnnotation!
        entity.setRefName(candidates.get(0).getTitle());
        entity.setRefId(candidates.get(0).getId());
        entity.setRefUrl(candidates.get(0).getUrl());
      }
      //log.trace("adding ner result: " + entity.getText() + " (" + entity.getBegin() + "," + entity.getLength() + ") with id " + entity.getRefId());
      entity.setSource(Annotation.Source.PRED);
      doc.addAnnotation(entity);
    }
  }
  
}
