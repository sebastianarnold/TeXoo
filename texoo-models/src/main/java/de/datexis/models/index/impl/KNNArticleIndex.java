package de.datexis.models.index.impl;

import de.datexis.common.Resource;
import de.datexis.models.index.ArticleRef;
import de.datexis.models.index.encoder.EntityEncoder;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import lombok.NonNull;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyHolder;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class KNNArticleIndex extends LuceneArticleIndex {

  protected final static Logger log = LoggerFactory.getLogger(KNNArticleIndex.class);
  
  protected ParagraphVectors parvec;
  EntityEncoder encoder;
      
  // these caches are used for nearest neighbour search (not efficient right now!)
  protected VocabCache<VocabWord> vocabCache;         // ids
  protected WeightLookupTable<VocabWord> lookupVectors  = null; // id -> mentionvec
  protected ModelUtils<VocabWord> lookupUtils;         // mentionvec -> id
  
  public KNNArticleIndex(Resource parVec) throws IOException {
    super();
    encoder = new EntityEncoder(parVec, EntityEncoder.Strategy.NAME);
    generateLookupCache();
  }
  
  protected void generateLookupCache() {
    log.debug("building entity list....");

    VocabularyHolder ids = new VocabularyHolder.Builder().build();
    
    vocabCache = new InMemoryLookupCache();
    lookupUtils = new BasicModelUtils<>();
    
    
    int num = 0;
    try {
      
      IndexReader reader = searcher.getIndexReader();
      List<ArticleRef> entries = new ArrayList<>(reader.maxDoc());
      
      for(int i=0; i<reader.maxDoc(); i++) {
        
        // read entity
        Document d = reader.document(i);
        ArticleRef ref = createWikidataArticleRef(d);
        
        // add all IDs
        String id = ref.getId();
        if(!ids.containsWord(id)) ids.addWord(id);
        else ids.incrementWordCounter(id);
        
      }
      
      ids.updateHuffmanCodes();
      ids.transferBackToVocabCache(vocabCache, true);

      // append vectors
      lookupVectors = new InMemoryLookupTable<>(vocabCache, encoder.getVectorSize(), true, 0.01, Nd4j.getRandom(), 0, true);
      lookupVectors.resetWeights();
      
      // create index of vectors
      num = 0;
      for(ArticleRef ref : entries) {
        INDArray embedding = encoder.encodeEntity(ref);
        lookupVectors.putVector(ref.getId(), embedding);
        if(++num % 100000 == 0) log.info("inserted " + num + " vectors into lookup table");
      }
      log.info("generated " + entries.size() + " entity vectors");
      
      lookupUtils.init(lookupVectors);

      // TODO: save utils / lookuptables into cache
      log.info("initialized lookup tables");
      
    } catch(IOException ex) {
      log.error(ex.toString());
    }
    
  }
  
  public void saveModel(Resource modelPath, String name) throws IOException {
    writeBinaryModel(lookupVectors, modelPath.resolve(name + "_lookup.bin").getOutputStream());
  }
  
  /**
   * Writes the model to DATEXIS binary format
   * @param vec
   * @param outputStream 
   */
  private static void writeBinaryModel(@NonNull WeightLookupTable<VocabWord> vec, @NonNull OutputStream outputStream) throws IOException {
    
    int words = 0;
    
    try(BufferedOutputStream buf = new BufferedOutputStream(outputStream);
         DataOutputStream writer = new DataOutputStream(buf)) {
      for(String word : vec.getVocabCache().words()) {
        if(word == null) continue;
        INDArray wordVector = vec.vector(word);
        log.trace("Write: " + word + " (size " + wordVector.length() + ")");
        writer.writeUTF(word);
        Nd4j.write(wordVector, writer);
        words++;
      }
      writer.flush();
    }
    
    log.info("Wrote " + words + " words with size " + vec.layerSize());
    
  }
  
  public List<ArticleRef> querySimilarArticles(String wikidataId, int hits) {
    ArrayList<ArticleRef> result = new ArrayList<>(hits);
    for(String id : lookupUtils.wordsNearest(wikidataId, hits)) {
      Optional<ArticleRef> a = queryWikidataID(id);
      if(a.isPresent()) result.add(a.get());
    }
    return result;
  }
  
  public List<ArticleRef> querySimilarArticles(String entityName, String context, int hits) {
    ArrayList<ArticleRef> result = new ArrayList<>(hits);
    INDArray eVec = encoder.encode(entityName);
    INDArray cVec = encoder.encode(context);
    for(String id : lookupUtils.wordsNearest(Nd4j.hstack(eVec, cVec), hits)) {
      Optional<ArticleRef> a = queryWikidataID(id);
      if(a.isPresent()) result.add(a.get());
    }
    return result;
  }
  
  public List<ArticleRef> querySimilarArticles(INDArray vec, int hits) {
    ArrayList<ArticleRef> result = new ArrayList<>(hits);
    for(String id : lookupUtils.wordsNearest(vec, hits)) {
      Optional<ArticleRef> a = queryWikidataID(id);
      if(a.isPresent()) result.add(a.get());
    }
    return result;
  }
  
}
