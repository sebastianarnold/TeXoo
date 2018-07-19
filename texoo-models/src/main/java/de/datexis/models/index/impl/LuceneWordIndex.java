package de.datexis.models.index.impl;

import de.datexis.common.Resource;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.models.index.WordIndex;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.KeywordTokenizerFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.icu.ICUFoldingFilterFactory;
import org.apache.lucene.analysis.icu.segmentation.ICUTokenizerFactory;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.RAMDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LuceneWordIndex extends LuceneIndex implements WordIndex {

  protected final static Logger log = LoggerFactory.getLogger(LuceneWordIndex.class);
  
  protected final static String FIELD_WORDS = "words";
  protected final static String FIELD_TEXT = "text";
  protected final static String FIELD_ID = "id";

  protected final static String PARAM_PROXIMITY = "2";
  protected final static String PARAM_FUZZY = "0.8";
  protected final static int NUM_CANDIDATES = 1000; // number of candidates to generate before scoring
  
  public LuceneWordIndex() {
  }
  
  public LuceneWordIndex(Iterable<String> texts) {
    createIndexRAM(texts);
  }
  
  public LuceneWordIndex(LookupCacheEncoder encoder) {
    createIndexRAM(encoder.getWords());
  }
  
  /**
   * Creates and opens a new index in local RAM
   */
  public void createIndexRAM(Iterable<String> texts) {
    try {
      RAMDirectory index = new RAMDirectory();
      createIndex(texts, index);
      openIndex(index);
    } catch (IOException e) {
      log.error(e.toString());
    }
  }
  
  public void createIndexDirectory(Iterable<String> texts, Resource cacheDir) throws IOException {
    FSDirectory index = FSDirectory.open(cacheDir.getPath());
    createIndex(texts, index);
    openIndex(index);
  }
  
  public void createIndex(Iterable<String> texts, Directory index) {
    log.info("creating new WordIndex...");
     
    analyzer = buildAnalyzer();
    
    try {

      IndexWriterConfig config = new IndexWriterConfig(analyzer);
      IndexWriter writer = new IndexWriter(index, config);
      
      int num = 0;
      int empty = 0;

      // articles as documents
      log.info("writing words...");
      for(String text : texts) {
        Document doc = createLuceneDocument(text);
        writer.addDocument(doc);
        if(++num % 100000 == 0) log.info("wrote " + num + " entries so far");
      }

      writer.close();

      log.info(num + " texts (" + empty + " empty) written to index");
      
    } catch (IOException e) {
      log.error(e.toString());
    }
    
  }
  
  /**
   * Retrieve candidates for a query on the "text" field. All given words have to match (lowercase, any order).
   */
  @Override
  public List<String> queryText(String text, int hits) {
    
    try {
      Query exactQ    = new QueryParser(FIELD_WORDS, analyzer).parse("\"" + text + "\"~" + PARAM_PROXIMITY);
      return queryIndex(exactQ, hits);
    } catch(Exception ex) {
      log.error(ex.toString());
    }
    
    return new ArrayList<>();
  }
  
  /**
   * Retrieve candidates for a query on the "text" field. Complete text has to match (lowercase, correct order).
   */
  @Override
  public List<String> queryExactText(String text, int hits) {
    
    try {
      Query exactQ    = new QueryParser(FIELD_TEXT, analyzer).parse("\"" + text + "\"");
      return queryIndex(exactQ, hits);
    } catch(Exception ex) {
      log.error(ex.toString());
    }
    
    return new ArrayList<>();
  }

  /**
   * Retrieve candidates for auto completion on the "text" field.
   */
  @Override
  public List<String> queryPrefixText(String prefix, int hits) {
    
    try {
      //log.info(prefix);
      prefix = prefix.replaceAll("\\s+", "\\\\ ");
      //log.info(prefix);
      Query prefixQ    = //new PrefixQuery(new Term(FIELD_TEXT, prefix));
          new QueryParser(FIELD_TEXT, analyzer).parse("" + prefix + "*");
      log.info(prefixQ.toString());
      return queryIndex(prefixQ, hits);
    } catch(Exception ex) {
      log.error(ex.toString());
    }
    
    return new ArrayList<>();
    
  }

  private Document createLuceneDocument(String text) {
    Document doc = new Document();
    addTextField(doc, FIELD_TEXT, text.trim(), Field.Store.YES);
    addTextField(doc, FIELD_WORDS, text.trim(), Field.Store.NO);
    return doc;
  }
  
  
  
  @Override
  protected Analyzer buildAnalyzer() {
    
    Map<String,Analyzer> analyzers = new TreeMap<>();
    
    try {

      // tokenize words, lowercase and standardize unicode characters
      Analyzer wordAnalyzer = CustomAnalyzer.builder()
            .withTokenizer(ICUTokenizerFactory.class)
            .addTokenFilter(ICUFoldingFilterFactory.class)
            //.addTokenFilter(StandardFilterFactory.class)
            //.addTokenFilter(LowerCaseFilterFactory.class)
            //.addTokenFilter(StopFilterFactory.class, "ignoreCase", "false", "words", "stopwords.txt", "format", "wordset")
            .build();
      // do not tokenize, lowercase and standardize unicode characters
      Analyzer stringAnalyzer = CustomAnalyzer.builder()
            .withTokenizer(KeywordTokenizerFactory.class)
            .addTokenFilter(ICUFoldingFilterFactory.class)
            //.addTokenFilter(StandardFilterFactory.class)
            //.addTokenFilter(LowerCaseFilterFactory.class)
            //.addTokenFilter(StopFilterFactory.class, "ignoreCase", "false", "words", "stopwords.txt", "format", "wordset")
            .build();
      analyzers.put(FIELD_WORDS, wordAnalyzer);
      analyzers.put(FIELD_TEXT, stringAnalyzer);

    } catch (IOException e) {
      log.error("Could not create Lucene Analyzer: ");
      log.error(e.toString());
    }
    
    return new PerFieldAnalyzerWrapper(new StandardAnalyzer(), analyzers);
    
  }
  
  
  protected List<String> queryIndex(Query query, int hits) {
    
    List<String> result = new ArrayList<>(hits);
    
    try {
      TopDocs top = searcher.search(query, hits);
      //log.info(q.toString());
      ScoreDoc[] docs = top.scoreDocs;
      for(ScoreDoc hit : docs) {
        Document d = searcher.doc(hit.doc);
        result.add(d.get(FIELD_TEXT));
      }
      
    } catch(Exception ex) {
      log.error(ex.toString());
    }
    
    return result;
    
  }
  
}
