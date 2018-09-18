package de.datexis.index;

import de.datexis.index.impl.LuceneArticleIndex;
import de.datexis.common.Resource;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.util.stream.Stream;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.datexis.index.impl.KNNArticleIndex;
import de.datexis.index.impl.VectorArticleIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ArticleIndexFactory {

  protected final static Logger log = LoggerFactory.getLogger(ArticleIndexFactory.class);
  
  /**
   * Loads a Wikidata Index from file into In-Memory Lucene instance
   */
  public static LuceneArticleIndex loadWikiDataIndex(Resource file) throws IOException {
    return loadWikiDataIndex(file, null, null, LuceneArticleIndex.class);
  }
  
  /**
   * Loads a Wikidata Index from file into Lucene instance cached on disk
   */
  public static LuceneArticleIndex loadWikiDataIndex(Resource file, Resource cacheDir) throws IOException {
    return loadWikiDataIndex(file, null, cacheDir, LuceneArticleIndex.class);
  }

  /**
   * Loads a Wikidata Index from file into Lucene instance cached on disk with ParVec embeddings
   */
  public static VectorArticleIndex loadWikiDataIndex(Resource file, Resource parvec, Resource cacheDir) throws IOException {
    return loadWikiDataIndex(file, parvec, cacheDir, VectorArticleIndex.class);
  }
  
  /**
   * Loads a Wikidata Index from file into In-Memory Lucene instance, uses cached instance if exists
   */
  protected static <T extends ArticleIndex> T loadWikiDataIndex(Resource file, Resource parvec, Resource cacheDir, Class<T> type) throws IOException {
    
    log.info("Reading WikiData index from " + file.toString());
    try(InputStream in = file.getInputStream()) {
      
      LuceneArticleIndex index = null;
      if(type.equals(VectorArticleIndex.class)) index = new VectorArticleIndex(parvec);
      else if(type.equals(KNNArticleIndex.class)) index = new KNNArticleIndex(parvec);
      else index = new LuceneArticleIndex();
      // try to load existing index from cache
      String cacheName = file.getFileName().replaceFirst("\\.json(\\.gz)?$", "");
      if(type.equals(VectorArticleIndex.class)) cacheName += "+parvec";
      if(cacheDir != null && index.openIndex(cacheDir.resolve("/" + cacheName + "/"))) {
        return (T)index;
      } else {
        // write new index to cache or RAM
        CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
        BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
        Stream<WikiDataArticle> articles = br.lines()
          .filter(line -> !line.isEmpty())
          .map(json -> convert(json))
          .filter(obj -> obj != null);
        if(cacheDir != null) index.createIndexDirectory(articles.iterator(), cacheDir.resolve("/" + cacheName + "/"));
        else index.createIndexRAM(articles.iterator());
        return (T)index;
      }
    }
    
  }

  private static ObjectMapper mapper = new ObjectMapper();
  
  private static WikiDataArticle convert(String json) {
    try {
      return mapper.readerFor(WikiDataArticle.class).readValue(json);
    } catch (IOException ex) {
      log.warn("Could not parse JSON: " + ex.toString());
    }
    return null;
  }
  

}
