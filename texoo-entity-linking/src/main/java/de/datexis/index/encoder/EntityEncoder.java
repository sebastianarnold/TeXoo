package de.datexis.index.encoder;

import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.index.ArticleRef;
import de.datexis.index.WikiDataArticle;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Span;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EntityEncoder extends Encoder {

  protected final static Logger log = LoggerFactory.getLogger(EntityEncoder.class);

  public static enum Strategy { NAME, NAME_CONTEXT };
  
  protected ParagraphVectors parvec;
  protected Strategy strategy;
  
  public EntityEncoder(Resource paragraphVectors, Strategy strategy) throws IOException {
    loadModel(paragraphVectors);
    this.strategy = strategy;
  }
  
  @Override
  public void loadModel(Resource paragraphVectors) throws IOException {
    log.info("loading paragraph vectors...");
    parvec = WordVectorSerializer.readParagraphVectors(paragraphVectors.getInputStream());
    TokenizerFactory t = new DefaultTokenizerFactory();
    t.setTokenPreProcessor(new MinimalLowercasePreprocessor());
    parvec.setTokenizerFactory(t);
    log.info("loaded " + parvec.getLabelsSource().getLabels().size() + " paragraph labels with size " + parvec.getLayerSize());
  }
  
  @Override
  public long getEmbeddingVectorSize() {
    if(strategy.equals(Strategy.NAME)) return parvec.getLayerSize();
    else if(strategy.equals(Strategy.NAME_CONTEXT)) return parvec.getLayerSize() * 2;
    else throw new IllegalArgumentException("invalid strategy");
  }

  public INDArray encodeEntity(WikiDataArticle art) {
    return encodeEntity(art.getId(), art.getTitle(), art.getDescription());
  }
  
  public INDArray encodeEntity(ArticleRef ref) {
    return encodeEntity(ref.getId(), ref.getTitle(), ref.getDescription());
  }
  
  private INDArray encodeEntity(String id, String title, String description) {
    INDArray nameEmbedding = encodeID(id, title);
    if(strategy.equals(Strategy.NAME)) {
      //if(nameEmbedding.sumNumber().intValue() == 0) nameEmbedding = contextEmbedding;
      return nameEmbedding;
    } else if(strategy.equals(Strategy.NAME_CONTEXT)) {
      // TODO: deactivated because results were bad
      String context = title;
      if(description != null) context += " " + description;
      INDArray contextEmbedding = encode(context);
      if(contextEmbedding.maxNumber().doubleValue() == 0) contextEmbedding = nameEmbedding;
      return Nd4j.hstack(nameEmbedding, contextEmbedding);
    } else {
      throw new IllegalArgumentException("invalid strategy");
    }
  }
  
  public INDArray encodeID(String id, String fallback) {
    try {
      return normalize(parvec.getWordVectorMatrix(id));
    } catch(Exception e) { // no matching label in model
      return null;//Nd4j.zeros(parvec.getLayerSize());
      // TODO: deactivated because results were bad [EVALUATE]
      //return encode(fallback);
    }
  }
  
  public INDArray encodeMention(String mention, String context) {
    INDArray nameEmbedding = encode(mention);
    if(strategy.equals(Strategy.NAME)) {
      return nameEmbedding;
    } else if(strategy.equals(Strategy.NAME_CONTEXT)) {
      INDArray contextEmbedding = encode(context);
      return Nd4j.hstack(nameEmbedding, contextEmbedding);
    } else {
      throw new IllegalArgumentException("invalid strategy");
    }
  }
  
  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }

  @Override
  public INDArray encode(String word) {
    try {
      return normalize(parvec.inferVector(word));
    } catch(Exception e) { // no matching words in model
      return Nd4j.zeros(parvec.getLayerSize());
    }
  }

  /**
   * Encodes each annotation in the document and attaches the vector to it.
   */
  public void encodeEach(Document doc, Annotation.Source source, Class<? extends Annotation> type) {
    doc.streamAnnotations(source, type).forEach(ann -> {
        String entityMention = ann.getText();
        String entityContext = doc.getSentenceAtPosition(ann.getBegin()).get().toTokenizedString();
          INDArray vec = encodeMention(entityMention, entityContext);
          ann.putVector(EntityEncoder.class, vec);
    });
  }
  
  private INDArray normalize(INDArray vec) {
    return vec != null ? Transforms.unitVec(vec) : null;
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public void saveModel(Resource dir, String name) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

}
