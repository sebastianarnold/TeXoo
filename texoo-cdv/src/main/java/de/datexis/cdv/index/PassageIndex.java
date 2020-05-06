package de.datexis.cdv.index;

import de.datexis.cdv.retrieval.EntityAspectQueryAnnotation;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Result;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.retrieval.model.RelevanceResult;
import de.datexis.retrieval.model.ScoredResult;
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
import java.util.stream.Collectors;

/**
 * Lucene Index that stores all Passages in a given dataset in memory.
 * Used as a retrieval baseline and for candidate generation.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class PassageIndex {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected final static String FIELD_DOCID = "docId";
  protected final static String FIELD_PARID = "parId";
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
      analyzers.put(FIELD_PARID, idAnalyzer);
      
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
    
    IndexWriter passageWriter = new IndexWriter(indexDir, config);
    
    int num = 0;
    
    // articles as documents
    log.info("writing passages...");
    for(de.datexis.model.Document doc : data.getDocuments()) {
      List<PassageAnnotation> anns = doc
        .streamAnnotations(Annotation.Source.GOLD, PassageAnnotation.class, true)
        .sorted()
        .collect(Collectors.toList());
      for(PassageAnnotation ann : anns) {
        // add article
        if(ann.getLength() < 10) continue;
        Document p = new Document();
        p.add(new StringField(FIELD_DOCID, doc.getId(), Field.Store.YES));
        p.add(new StringField(FIELD_PARID, ann.getId(), Field.Store.YES));
        p.add(new TextField(FIELD_TEXT, doc.getText(ann), Field.Store.NO));
        passageWriter.addDocument(p);
        if(++num % 100000 == 0) log.info("wrote " + num + " passages so far");
      }
    }
    
    passageWriter.close();
    
    log.info("{} passages written to index", num);
    
  }
  
  /**
   * Add Lucene candidates to Queries as GOLD RelevanceResult
   */
  public void retrievePassageCandidates(Dataset corpus, int num) {
    
    Random random = new Random();
    for(de.datexis.model.Query q : corpus.getQueries()) {
      String entity = q.getAnnotation(EntityAspectQueryAnnotation.class).getEntity();
      String aspect = q.getAnnotation(EntityAspectQueryAnnotation.class).getAspectHeading();
      if(aspect == null) aspect = q.getAnnotation(EntityAspectQueryAnnotation.class).getAspect();
      entity = entity.replace("/", " ").replace("-", " ").replace(":", " ");
      aspect = aspect.replace("/", " ").replace("-", " ").replace(":", " ");
      
      List<PassageResult> candidates = search(entity + " " + aspect, num);
      if(candidates.size() == 0)
        log.error("no match found");
  
      // append all passages of the first retrieved document (in case of exact match)
      Optional<de.datexis.model.Document> doc = corpus.getDocument(candidates.get(0).documentId);
      if(!doc.isPresent())
        log.error("Document not found: {}", candidates.get(0).documentId);
      int i = 1;
      for(PassageAnnotation ann : doc.get().streamAnnotations(Annotation.Source.GOLD, PassageAnnotation.class, true).collect(Collectors.toList())) {
        candidates.add(i++, new PassageResult(doc.get().getId(), ann.getId(), 0));
        if(ann.getLength() < 10) log.warn("Found very short passage {} in document {}", ann.getId(), ann.getDocumentRef().getId());
      }
      
      // fill up with up to NUM random candidate passages
      /*while(candidates.size() < num) {
        de.datexis.model.Document doc = corpus.getRandomDocument().get();
        List<PassageAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, PassageAnnotation.class, true).collect(Collectors.toList());
        int idx = random.nextInt(anns.size());
        PassageAnnotation ann = anns.get(idx);
        PassageResult candidate = new PassageResult(doc.getId(), ann.getId(), 0);
        if(!candidates.contains(candidate)) candidates.add(candidate);
      }*/
  
      //log.info("Adding {} + {} results for query '{}'", n, (candidates.size() - n), entity);
      for(PassageResult candidate : candidates) {
        if(q.getResults().size() >= num) break;
        doc = corpus.getDocument(candidate.documentId);
        if(!doc.isPresent())
          log.error("Document not found: {}", candidate.documentId);
        boolean found = false;
        for(Result r : q.getResults()) {
          // check if this candidate was already included in results
          if(r.getId().equals(candidate.passageId)) {
            found = true;
            break;
          }
        }
        if(!found) {
          for(PassageAnnotation ann : doc.get().streamAnnotations(Annotation.Source.GOLD, PassageAnnotation.class, true).collect(Collectors.toList())) {
            if(ann.getId().equals(candidate.passageId)) {
              found = true;
              if(ann.getLength() < 10) {
                log.warn("Found very short passage {} in document {}", ann.getId(), ann.getDocumentRef().getId());
                continue;
              }
              RelevanceResult resultAnnotation = new RelevanceResult(Annotation.Source.SILVER, doc.get(), ann.getBegin(), ann.getEnd());
              resultAnnotation.setRelevance(0);
              resultAnnotation.setId(ann.getId());
              resultAnnotation.setDocumentRef(doc.get());
              q.addResult(resultAnnotation);
              break;
            }
          }
        }
        if(!found)
          log.error("passage not found: {}", candidate.passageId);
      }
    }
  }
  
  /**
   * Add Lucene results to Queries as PRED ScoredResult
   * @param candidatesOnly set to TRUE to restrict search on given candidates only
   */
  public void retrieveAllQueries(Dataset corpus, int hits, boolean candidatesOnly) {
    
    for(de.datexis.model.Query q : corpus.getQueries()) {
      String entity = q.getAnnotation(EntityAspectQueryAnnotation.class).getEntity();
      String aspect = q.getAnnotation(EntityAspectQueryAnnotation.class).getAspectHeading();
      if(aspect == null) aspect = q.getAnnotation(EntityAspectQueryAnnotation.class).getAspect();
      entity = entity.replace("/", " ").replace("-", " ").replace(":", " ");
      
      List<RelevanceResult> candidates = q.getResults(Annotation.Source.GOLD, RelevanceResult.class);
      
      List<PassageResult> results = search(entity + " " + aspect, hits);
      if(results.size() == 0)
        log.error("no match found");
      
      for(PassageResult result : results) {
        Optional<de.datexis.model.Document> doc = corpus.getDocument(result.documentId);
        if(!doc.isPresent())
          log.error("Document not found: {}", result.documentId);
        boolean found = false;
        for(Result c : candidates) {
          // check if this result was included in candidates
          if(c.getId().equals(result.passageId)) found = true;
        }
        if(found || !candidatesOnly) {
          found = false;
          for(PassageAnnotation ann : doc.get().streamAnnotations(Annotation.Source.GOLD, PassageAnnotation.class, true).collect(Collectors.toList())) {
            if(ann.getId().equals(result.passageId)) {
              found = true;
              ScoredResult resultAnnotation = new ScoredResult(Annotation.Source.PRED, doc.get(), ann.getBegin(), ann.getEnd());
              resultAnnotation.setRelevance(1);
              resultAnnotation.setScore(result.score);
              resultAnnotation.setId(ann.getId());
              resultAnnotation.setDocumentRef(doc.get());
              q.addResult(resultAnnotation);
              break;
            }
          }
          if(!found)
            log.error("candidate passage not found: {}", result.passageId);
        }
      }
    }
  }
  
  /**
   *  @return a list of <document,passage> IDs
   */
  public List<PassageResult> search(String querystring, int hits) {
    
    List<PassageResult> result = new ArrayList<>();
    
    try {
      Query query = new QueryParser(FIELD_TEXT, analyzer).parse(querystring);//.parse("\"" + name + "\"");
      TopDocs top = searcher.search(query, hits);
      ScoreDoc[] docs = top.scoreDocs;
      for(ScoreDoc hit : docs) {
        Document d = searcher.doc(hit.doc);
        result.add(new PassageResult(d.get(FIELD_DOCID), d.get(FIELD_PARID), hit.score));
      }
      
    } catch(ParseException e) {
      log.error(e.toString());
    } catch(IOException e) {
      e.printStackTrace();
    }
    
    return result;
    
  }
  
  public class PassageResult {
    public String documentId;
    public String passageId;
    public double score;
    public PassageResult(String docId, String parId, float score) {
      this.documentId = docId;
      this.passageId = parId;
      this.score = score;
    }
  
    @Override
    public boolean equals(Object o) {
      if(this == o) return true;
      if(o == null || getClass() != o.getClass()) return false;
      PassageResult that = (PassageResult) o;
      return documentId.equals(that.documentId) &&
        passageId.equals(that.passageId);
    }
  
    @Override
    public int hashCode() {
      return Objects.hash(documentId, passageId);
    }
  }
  
}
