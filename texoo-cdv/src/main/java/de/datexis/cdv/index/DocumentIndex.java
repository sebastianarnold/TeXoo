package de.datexis.cdv.index;

import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.icu.ICUFoldingFilterFactory;
import org.apache.lucene.analysis.icu.segmentation.ICUTokenizerFactory;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.BaseDirectory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.RAMDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

/**
 * Lucene Index that stores all Documents in a given dataset in memory.
 * Used for candidate generation.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class DocumentIndex {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected final static String FIELD_DOCID = "docId";
  protected final static String FIELD_TEXT = "text";
  
  public final static int NUM_CANDIDATES = 64;
  
  protected IndexReader reader;
  protected IndexSearcher searcher;
  protected Analyzer analyzer;
  
  public void loadModel(Resource path) throws IOException {
    FSDirectory index = FSDirectory.open(path.getPath());
    reader = DirectoryReader.open(index);
    searcher = new IndexSearcher(reader);
    analyzer = buildAnalyzer();
  }
  
  /**
   * Build analysers for id and text fields.
   */
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
      analyzers.put(FIELD_TEXT, wordAnalyzer);
      
      // use whole string as token
      Analyzer idAnalyzer = new KeywordAnalyzer();
      analyzers.put(FIELD_DOCID, idAnalyzer);
      
    } catch (IOException e) {
      log.error("Could not create Lucene Analyzer: ");
      log.error(e.toString());
    }
    
    return new PerFieldAnalyzerWrapper(new StandardAnalyzer(), analyzers);
    
  }
  
  public void createInMemoryIndex(Dataset data) throws IOException {
    RAMDirectory indexDir = new RAMDirectory();
    createIndex(data, indexDir);
    reader = DirectoryReader.open(indexDir);
    searcher = new IndexSearcher(reader);
    analyzer = buildAnalyzer();
  }
  
  public void setSimilarity(Similarity sim) {
    searcher.setSimilarity(sim);
  }
  
  public void createIndex(Dataset data, Resource indexPath) throws IOException {
    log.info("creating new passage index in path '{}'...", indexPath.toString());
    FSDirectory indexDir = FSDirectory.open(indexPath.getPath());
    createIndex(data, indexDir);
  }
  
  /**
   * Create a new index from all passages in the given dataset.
   */
  protected void createIndex(Dataset data, BaseDirectory indexDir) throws IOException {
    
    analyzer = buildAnalyzer();
    IndexWriterConfig config = new IndexWriterConfig(analyzer);
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    config.setCommitOnClose(true);
    config.setSimilarity(new BM25Similarity());
    
    IndexWriter documentWriter = new IndexWriter(indexDir, config);
    
    int num = 0;
    
    // articles as documents
    log.info("writing documents...");
    for(de.datexis.model.Document doc : data.getDocuments()) {
      // add article
      Document p = new Document();
      p.add(new StringField(FIELD_DOCID, doc.getId(), Field.Store.YES));
      p.add(new TextField(FIELD_TEXT, doc.getText(), Field.Store.NO));
      documentWriter.addDocument(p);
      if(++num % 100000 == 0) log.info("wrote " + num + " documents so far");
    }
    
    documentWriter.close();
    
    log.info("{} documents written to index", num);
    
  }
  
  /**
   *  @return a list of <document,passage> IDs
   */
  public List<DocumentResult> search(String querystring, int hits) {
    
    List<DocumentResult> result = new ArrayList<>();
    
    try {
      Query query = new QueryParser(FIELD_TEXT, analyzer).parse(querystring);
      TopDocs top = searcher.search(query, hits);
      ScoreDoc[] docs = top.scoreDocs;
      for(ScoreDoc hit : docs) {
        Document d = searcher.doc(hit.doc);
        result.add(new DocumentResult(d.get(FIELD_DOCID), hit.score));
      }
      
    } catch(ParseException e) {
      log.error(e.toString());
    } catch(IOException e) {
      e.printStackTrace();
    }
    
    return result;
    
  }
  
  public class DocumentResult {
    public String documentId;
    public double score;
    public DocumentResult(String docId, float score) {
      this.documentId = docId;
      this.score = score;
    }
  
    @Override
    public boolean equals(Object o) {
      if(this == o) return true;
      if(o == null || getClass() != o.getClass()) return false;
      DocumentResult that = (DocumentResult) o;
      return documentId.equals(that.documentId);
    }
  
    @Override
    public int hashCode() {
      return Objects.hash(documentId);
    }
  }
  
}
