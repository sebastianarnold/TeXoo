package de.datexis.index.impl;

import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.index.ArticleRef;
import de.datexis.index.WikiDataArticle;
import de.datexis.index.encoder.EntityEncoder;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.lucene.document.Document;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class VectorArticleIndex extends LuceneArticleIndex {

  protected final static Logger log = LoggerFactory.getLogger(VectorArticleIndex.class);
  
  protected final static int NUM_PARVEC_CANDIDATES = 512; // number of candidates to score with cosine similarity
  protected final static EntityEncoder.Strategy strategy = EntityEncoder.Strategy.NAME_CONTEXT;
      
  EntityEncoder encoder;
      
  public VectorArticleIndex(Resource parVec) throws IOException {
    super();
    encoder = new EntityEncoder(parVec, EntityEncoder.Strategy.NAME);
  }
  
  /**
   * Adds a Vector field to Lucene index
   */
  @Override
  protected Document createLuceneDocument(WikiDataArticle article) {
    Document doc = super.createLuceneDocument(article);
    // generate vectors
    INDArray embedding = encoder.encodeEntity(article);
    if(embedding != null) addVectorField(doc, FIELD_VECTOR, embedding); // might encode Null-vector here
    else log.error("Could not encode entity {}", article.toString());
    return doc;
  }
  
  public List<ArticleRef> querySimilarArticles(String mention, String context, int hits) {
    ArrayList<ArticleRef> result = new ArrayList<>(hits);
    // first step: lucene query
    List<ArticleRef> candidates = queryNames(mention, NUM_PARVEC_CANDIDATES);
    // second step: reranking
    INDArray embedding = encoder.encodeMention(mention, context);
    for(ArticleRef ref : candidates) {
      INDArray entityEmbedding = ref.getVector();
      if(entityEmbedding != null) {
        INDArray candidate = Nd4j.hstack(entityEmbedding, entityEmbedding); // FIXME: only needed because we skipped context in EntityEncoder
        double score = WordHelpers.cosineSim(candidate, embedding);
        if(Double.isFinite(score)) ref.setScore(score);
        else ref.setScore(0.);
      } else {
        //log.warn("missing vector for entity {}", ref.getId());
        ref.setScore(0.);
      }
      result.add(ref);
    }
    Collections.sort(result, new ArticleRef.ScoreComparator()); // this sort is stable, so it keeps Lucene sorting in case of score = 0
    /*for(ArticleRef ref : result) {
      log.debug(mention + "\t" + ref.toString() + "\t" + ref.getScore());
    }*/
    return result;
  }
  
}
