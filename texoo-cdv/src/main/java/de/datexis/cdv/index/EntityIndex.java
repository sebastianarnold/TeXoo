package de.datexis.cdv.index;

import de.datexis.cdv.model.EntityAspectAnnotation;
import de.datexis.cdv.retrieval.EntityAspectQueryAnnotation;
import de.datexis.encoder.IEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Query;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.retrieval.index.InMemoryIndex;
import de.datexis.retrieval.preprocess.WikipediaUrlPreprocessor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * In index that holds entities (e.g. "Diabetes_mellitus_type_1") as keys and vectors as values.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EntityIndex extends QueryIndex {
  
  protected final static Logger log = LoggerFactory.getLogger(EntityIndex.class);
  
  /** separator used for multi-label ids */
  public static String ID_SEPARATOR_REGEX = ";";
  
  /** called from JSON deserialization */
  protected EntityIndex() {}
  
  public EntityIndex(IEncoder encoder) {
    super(new WikipediaUrlPreprocessor(), encoder);
    this.id = "ENT";
  }
  
  /**
   * Lookup an Entity ID into a dense vector. Will average over multiple IDs seperated with ";".
   */
  @Override
  public INDArray lookup(String id) {
    String[] ids = id.split(ID_SEPARATOR_REGEX);
    INDArray vec, sum = Nd4j.zeros(DataType.FLOAT, getEmbeddingVectorSize(), 1);
    int count = 0;
    for(String s : ids) {
      vec = super.lookup(s);
      if(vec != null) {
        sum.addi(vec);
        count++;
      }
    }
    if(count == 0) return null;
    else return count > 1 ? sum.divi(count) : sum;
  }
  
  /**
   * Encode an entity into a dense vector.
   */
  @Override
  public INDArray encode(String mention) {
    return super.encode(mention.replace('_', ' '));
  }
  
  /**
   * Encode a entity into a sparse vector.
   */
  public INDArray decode(String entityId) {
    INDArray result = Nd4j.zeros(DataType.FLOAT, size(), 1);
    String[] ids = entityId.split(ID_SEPARATOR_REGEX);
    for(String id : ids) {
      int idx = index(id);
      if(idx >= 0) result.putScalarUnsafe(idx, 1.);
    }
    if(result.sumNumber().doubleValue() == 0.)
      log.warn("entity '{}' not contained in index", entityId);
    return result;
  }
  
  /**
   * Build an index over all entities given by GOLD EntityAspectAnnotations in the Documents.
   */
  @Override
  @Deprecated
  public void trainModel(Collection<Document> documents) {
    List<String> keys = new ArrayList<>();
    Map<String, INDArray> vectors = new HashMap<>();
    for(Document doc : documents) {
      for(EntityAspectAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, EntityAspectAnnotation.class)) {
        //String key = keyPreprocessor.preProcess(q.get);
        String focusId = ann.getEntityId();
        Sentence focus = DocumentFactory.createSentenceFromTokenizedString(ann.getEntity());
        if(focusId != null) {
          String[] ids = focusId.split(ID_SEPARATOR_REGEX);
          for(String key : ids) {
            key = keyPreprocessor.preProcess(key);
            keys.add(key);
            if(!vectors.containsKey(key)) {
              INDArray vec = encoder.encode(focus);
              vectors.put(key, vec);
            }
          }
        }
      }
    }
    buildKeyIndex(keys, false);
    buildVectorIndex(vectors, false);
    setModelAvailable(true);
  }
  
  /**
   * Build the query index to include all query types
   * @param queries
   */
  public void encodeFromQueries(Collection<Query> queries) {
    List<String> keys = new ArrayList<>();
    Map<String, INDArray> vectors = new HashMap<>();
    for(Query q : queries) {
      EntityAspectQueryAnnotation ann = q.getAnnotation(EntityAspectQueryAnnotation.class);
      //String key = keyPreprocessor.preProcess(q.get);
      String focusId = ann.getEntityId();
      Sentence focus = DocumentFactory.createSentenceFromTokenizedString(ann.getEntity());
      if(focusId != null) {
        String[] ids = focusId.split(ID_SEPARATOR_REGEX);
        for(String id : ids) {
          keys.add(id);
          if(!vectors.containsKey(id)) {
            INDArray vec = null;
            if(encoder instanceof InMemoryIndex)
              vec = ((InMemoryIndex)encoder).lookup(id);
            if(vec == null) {
              log.info("Fallback encoding entity {} '{}'", ann.getEntityId(), ann.getEntity());
              vec = encoder.encode(focus);
            }
            vectors.put(id, vec);
          }
        }
      } else {
        log.warn("Found query without entityID for '{}' - skipping", q.getId(), ann.getEntity());
      }
    }
    buildKeyIndex(keys, true);
    buildVectorIndex(vectors, true);
    setModelAvailable(true);
  }
  
}
