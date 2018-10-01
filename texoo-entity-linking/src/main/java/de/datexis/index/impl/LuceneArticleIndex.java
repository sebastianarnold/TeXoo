package de.datexis.index.impl;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Article;
import de.datexis.index.ArticleIndex;
import de.datexis.index.ArticleRef;
import de.datexis.index.WikiDataArticle;
import de.datexis.index.encoder.EntityEncoder;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import info.debatty.java.stringsimilarity.JaroWinkler;
import info.debatty.java.stringsimilarity.interfaces.NormalizedStringSimilarity;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.icu.ICUFoldingFilterFactory;
import org.apache.lucene.analysis.icu.segmentation.ICUTokenizerFactory;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.RAMDirectory;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LuceneArticleIndex extends ArticleIndex {

  protected final static Logger log = LoggerFactory.getLogger(LuceneArticleIndex.class);
  
  protected final static String FIELD_TITLE = "title";
  protected final static String FIELD_TYPE = "type";
  protected final static String FIELD_REFIDS = "refID_";
  protected final static String FIELD_REFIDS_WIKIDATA = "refID_wikidata";
  protected final static String FIELD_REFIDS_WIKIPEDIA = "refID_wikipedia"; // every index stores just one wikipedia language, so we do not mix IDs
  //protected final static String FIELD_REFIDS_ENWIKI = "refID_enwiki";
  //protected final static String FIELD_REFIDS_DEWIKI = "refID_dewiki";
  protected final static String FIELD_REFIDS_FREEBASE = "refID_freebase";
  protected final static String FIELD_REFIDS_UMLS = "refID_umls";
  protected final static String FIELD_REFIDS_ICD10 = "refID_icd10";
  protected final static String FIELD_REFURLS_WIKIPEDIA = "refURL_wikipedia";
  
  protected final static String FIELD_TEXT = "text";
  protected final static String FIELD_DESCRIPTION = "description";
  protected final static String FIELD_NAMES = "name";
  protected final static String FIELD_TERMS = "term";
  protected final static String FIELD_VECTOR = "vector";

  protected final static String PARAM_PROXIMITY = "2";
  protected final static String PARAM_FUZZY = "0.8";
  protected final static int NUM_BM25_CANDIDATES = 1024; // number of candidates to generate before scoring
  
  protected IndexReader reader;
  protected IndexSearcher searcher;
  protected Analyzer analyzer;
  protected final NormalizedStringSimilarity sim = new JaroWinkler();
  protected final TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
  
  public LuceneArticleIndex() {
  }
  
  /**
   * tries to open an existing index from given path
   */
  public boolean openIndex(Resource path) {
    try {
      FSDirectory index = FSDirectory.open(path.getPath());
      return openIndex(index);
    } catch (IOException e) {
      return false;
    }
  }
  
  private boolean openIndex(Directory index) {
    try {
      reader = DirectoryReader.open(index);
      searcher = new IndexSearcher(reader);
      //searcher.setSimilarity(new BM25Similarity(1f,1f));
      //searcher.setSimilarity(new ContextSimilarity());
      analyzer = buildAnalyzer();
      return true;
    } catch (IOException e) {
      return false;
    }
  }
  
  /**
   * Creates and opens a new index in local RAM
   */
  public void createIndexRAM(Iterator<? extends Article> articles) {
    RAMDirectory index = new RAMDirectory();
    createIndex(articles, index);
    openIndex(index);
  }
  
  public void createIndexDirectory(Iterator<? extends Article> articles, Resource cacheDir) throws IOException {
    FSDirectory index = FSDirectory.open(cacheDir.getPath());
    createIndex(articles, index);
    openIndex(index);
  }
  
  public void createIndex(Iterator<? extends Article> articles, Directory index) {
    log.info("creating new index...");
     
    analyzer = buildAnalyzer();
    
    try {

      IndexWriterConfig config = new IndexWriterConfig(analyzer);
      IndexWriter writer = new IndexWriter(index, config);
      
      int num = 0;
      int empty = 0;

      // articles as documents
      log.info("writing articles...");
      while(articles.hasNext()) {
        WikiDataArticle article = (WikiDataArticle) articles.next();
        // add article
        Document doc = createLuceneDocument(article);
        writer.addDocument(doc);
        if(++num % 100000 == 0) log.info("wrote " + num + " articles so far");
      }

      writer.close();

      log.info(num + " articles (" + empty + " empty) written to index");
      
    } catch (IOException e) {
      log.error(e.toString());
    }
    
  }
  
  /**
   * Retrieve candidates for a query on the "name" field. All given words have to match (lowercase, any order).
   * @param name The name to search, e.g. "obama barack"
   * @param hits Max number of hits to generate
   * @return An List of matching ArticleRefs, e.g. Wiki URL of "Barack Obama"
   */
  @Override
  public List<ArticleRef> queryNames(String name, int hits) {
    List<Document> docs = queryIndexNames(name, NUM_BM25_CANDIDATES);
    List<ArticleRef> candidates = new ArrayList<>(NUM_BM25_CANDIDATES);
    for(Document d : docs) {
      //log.debug("found " + d.get(FIELD_TITLE) + "\t" + d.get(FIELD_REFIDS + "wikidata") + "\t");
      Article a = createWikidataArticle(d);
      double score = 0.;
      for(String title : a.getNames()) {
        double jaroSim = sim.similarity(preprocessor.preProcess(title), preprocessor.preProcess(name));
        if(jaroSim > score) score = jaroSim;
      }
      ArticleRef ref = new ArticleRef(a);
      ref.setScore(score);
      candidates.add(ref);
    }
    Collections.sort(candidates, new ArticleRef.ScoreComparator()); // stable sort
    return Lists.newArrayList(Iterables.limit(candidates, hits));
  }

  /**
   * Retrieve candidates for auto completion on the "name" field.
   * @param prefix
   * @param hits
   * @return 
   */
  @Override
  public List<ArticleRef> queryPrefixNames(String prefix, int hits) {
    return queryIndexPrefix(prefix, hits);
  }

  /**
   * Retrieve the article with a given ID
   * @param id The indexed ID, e.g. "Q64"
   * @return The ArticleRef, if exists
   */
  @Override
  public Optional<ArticleRef> queryID(String id) {
    return queryWikidataID(id);
  }
  
  /**
   * Retrieve the article with a given ID
   * @param id The Wikidata ID, e.g. "Q64"
   * @return The Wiki article
   */
  public Optional<ArticleRef> queryWikidataID(String id) {
    Optional<Document> d = queryIndexID(FIELD_REFIDS_WIKIDATA, id);
    if(d.isPresent()) {
      ArticleRef ref = createWikidataArticleRef(d.get());
      return Optional.of(ref);
    } else return Optional.empty();
  }
  
  /**
   * Retrieve the article with a given URL
   * @return The Wiki article
   */
  public Optional<ArticleRef> queryWikipediaURL(String url) {
    if(url == null || url.isEmpty()) return Optional.empty();
    else if(url.startsWith("http://"))   url = url.replaceFirst("http://", "https://");
    else if(!url.startsWith("https://")) url = "https://" + url;
    url = decodeWikiUrl(url);
    Optional<Document> d = queryIndexID(FIELD_REFURLS_WIKIPEDIA, url);
    if(d.isPresent()) {
      ArticleRef ref = createWikidataArticleRef(d.get());
      return Optional.of(ref);
    } else return Optional.empty();
  }
  
  /**
   * Retrieve the article with a given name
   * @return The Wiki article
   */
  public Optional<ArticleRef> queryWikipediaPage(String name) {
    name = decodeWikiUrl(name);
    Optional<Document> d = queryIndexID(FIELD_REFIDS_WIKIPEDIA, name);
    if(d.isPresent()) {
      ArticleRef ref = createWikidataArticleRef(d.get());
      return Optional.of(ref);
    } else return Optional.empty();
  }
  
  protected String decodeWikiUrl(String url) {
    try {
      url = URLDecoder.decode(url, "UTF-8"); // replace escaped chars
    } catch(UnsupportedEncodingException ex) {
      log.debug("could not decode URL '" + url + "'");
    }
    url = url.replace(" ","_").replaceFirst("#.+$", ""); // replace spaces and remove anchor
    return url;
  }
  
  public Collection<String> getAllArticleTitles() {
    return getAllFields(FIELD_TITLE);
  }
  
  public Collection<String> getAllArticleNames() {
    return getAllFields(FIELD_NAMES);
  }
  
  public Collection<String> getAllArticleTerms() {
    return getAllFields(FIELD_TERMS);
  }
  
  public Collection<String> getAllArticleURLs() {
    return getAllFields(FIELD_REFURLS_WIKIPEDIA);
  }
  
  public Collection<String> getAllArticleIDs() {
    return getAllFields(FIELD_REFIDS_WIKIDATA);
  }
  
  protected Collection<String> getAllFields(String field) {
    Set<String> result = new TreeSet<>();
    try {
      IndexReader reader = searcher.getIndexReader();
      Set<String> fields = new TreeSet<>();
      fields.add(field);
      for(int i=0; i<reader.maxDoc(); i++) {
        Document d = reader.document(i, fields);
        String[] values = d.getValues(field);
        for(String v : values) {
          result.add(v);
        }
      }
    } catch(Exception ex) {
      log.error(ex.toString());
    }
    return result;
  }
  
  protected Document createLuceneDocument(WikiDataArticle article) {
    Document doc = new Document();
    // Entity Metadata
    //String title = article.getTitle().replaceFirst("\\ \\(.*\\)$", "").trim(); // replace Disambiguations
    addTextField(doc, FIELD_TITLE, article.getTitle().trim(), Field.Store.YES);
    addTextField(doc, FIELD_TYPE, article.getType(), Field.Store.YES);
    //addTextField(doc, FIELD_TEXT, article.getText(), Field.Store.NO);
    addTextField(doc, FIELD_DESCRIPTION, article.getDescription(), Field.Store.YES);
    // Aliases and Search Terms
    for(String name : article.getNames()) addNameField(doc, FIELD_NAMES, name);
    for(String term : article.getTerms()) addNameField(doc, FIELD_TERMS, term);
    // Reference IDs
    addStringField(doc, FIELD_REFIDS_WIKIDATA, article.getRefID(WikiDataArticle.RefID.WIKIDATA));
    addStringField(doc, FIELD_REFIDS_FREEBASE, article.getRefID(WikiDataArticle.RefID.FREEBASE));
    addStringField(doc, FIELD_REFIDS_WIKIPEDIA, article.getRefID(WikiDataArticle.RefID.WIKIPEDIA));
    addStringField(doc, FIELD_REFIDS_UMLS, article.getRefID(WikiDataArticle.RefID.UMLS));
    addStringField(doc, FIELD_REFIDS_ICD10, article.getRefID(WikiDataArticle.RefID.ICD10));
    addStringField(doc, FIELD_REFURLS_WIKIPEDIA, article.getUrl());
    //addStringField(doc, FIELD_REFURLS + "image", article.getRefURL("image"));
    // popularity
    // vectors
    return doc;
  }
  
  protected ArticleRef createWikidataArticleRef(Document doc) {
    ArticleRef art = new ArticleRef();
    art.setTitle(doc.get(FIELD_TITLE));
    art.setType(doc.get(FIELD_TYPE));
    art.setDescription(doc.get(FIELD_DESCRIPTION));
    art.setId(doc.get(FIELD_REFIDS_WIKIDATA));
    art.setUrl(doc.get(FIELD_REFURLS_WIKIPEDIA));
    String vec = doc.get(FIELD_VECTOR);
    if(vec != null) art.setVector(ObjectSerializer.getArrayFromBase64String(vec));
    return art;
  }
  
  protected Article createWikidataArticle(Document doc) {
    Article art = new Article();
    art.setTitle(doc.get(FIELD_TITLE));
    art.setType(doc.get(FIELD_TYPE));
    art.setDescription(doc.get(FIELD_DESCRIPTION));
    art.setId(doc.get(FIELD_REFIDS_WIKIDATA));
    art.setUrl(doc.get(FIELD_REFURLS_WIKIPEDIA));
    for(IndexableField name : doc.getFields(FIELD_NAMES)) {
      art.addName(name.stringValue());
    }
    String vec = doc.get(FIELD_VECTOR);
    if(vec != null) art.setVector(ObjectSerializer.getArrayFromBase64String(vec));
    return art;
  }
  
  protected String splitString(String name, String suffix) {
    String[] parts = name.split("\\s");
    StringBuilder result = new StringBuilder();
    for(String part : parts) {
      if(result.length() > 0) result.append(" ");
      result.append(part).append(suffix);
    }
    return result.toString();
  }
  
  /**
   * A TextField is a tokenized field in Lucene.
   */
  protected void addTextField(Document doc, String name, String value, Field.Store store) {
    if(value != null) doc.add(new TextField(name, value, store));
  }
  
  /**
   * A StringField is a non-tokenized field in Lucene. We always store.
   */
  protected void addStringField(Document doc, String name, String value) {
    if(value != null) doc.add(new StringField(name, value, Field.Store.YES));
  }
  
  /**
   * A NameField is a tokenized lowercase field in Lucene. We always store.
   */
  protected void addNameField(Document doc, String name, String value) {
    if(value != null) doc.add(new Field(name, value, FIELDTYPE_NAME));
  }
  
  /**
   * A VectorField is a compressed INDArray as Base64 encoded string.
   */
  protected void addVectorField(Document doc, String name, INDArray arr) {
    String value = ObjectSerializer.getArrayAsBase64String(arr);
    if(value != null) doc.add(new Field(name, value, FIELDTYPE_VECTOR));
  }
  
  static final FieldType FIELDTYPE_NAME;
  static final FieldType FIELDTYPE_VECTOR;
  static {
    FIELDTYPE_NAME = new FieldType();
    FIELDTYPE_NAME.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    FIELDTYPE_NAME.setTokenized(true);
    FIELDTYPE_NAME.setStored(true); // required for string similarity
    FIELDTYPE_NAME.setOmitNorms(true);
    FIELDTYPE_NAME.freeze();
    FIELDTYPE_VECTOR = new FieldType();
    FIELDTYPE_VECTOR.setIndexOptions(IndexOptions.NONE);
    FIELDTYPE_VECTOR.setTokenized(false);
    FIELDTYPE_VECTOR.setStored(true); // required for vector retrieval
    FIELDTYPE_VECTOR.setOmitNorms(true);
    FIELDTYPE_VECTOR.freeze();
  }
  
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
      //analyzers.put(FIELD_TITLE, wordAnalyzer);
      analyzers.put(FIELD_NAMES, wordAnalyzer);
      analyzers.put(FIELD_TERMS, wordAnalyzer);
      
      // use whole string as token
      Analyzer idAnalyzer = new KeywordAnalyzer();
      analyzers.put(FIELD_VECTOR, idAnalyzer);
      analyzers.put(FIELD_REFIDS_WIKIDATA, idAnalyzer);
      analyzers.put(FIELD_REFIDS_WIKIPEDIA, idAnalyzer);
      analyzers.put(FIELD_REFIDS_FREEBASE, idAnalyzer);
      analyzers.put(FIELD_REFIDS_UMLS, idAnalyzer);
      analyzers.put(FIELD_REFIDS_ICD10, idAnalyzer);
      analyzers.put(FIELD_REFURLS_WIKIPEDIA, idAnalyzer);

    } catch (IOException e) {
      log.error("Could not create Lucene Analyzer: ");
      log.error(e.toString());
    }
    
    return new PerFieldAnalyzerWrapper(new StandardAnalyzer(), analyzers);
    
  }
  
  
  /*static final FieldType SIGNATURE_TYPE = new FieldType();
    SIGNATURE_TYPE.setIndexOptions(IndexOptions.NONE);
    SIGNATURE_TYPE.setTokenized(false);
    SIGNATURE_TYPE.setStored(true);
    SIGNATURE_TYPE.setOmitNorms(true);
    SIGNATURE_TYPE.freeze();*/

  protected List<Document> queryIndexNames(String name, int hits) {
    
    List<Document> result = new ArrayList<>();
    
    try {
      Query exactQ    = new BoostQuery(new QueryParser(FIELD_NAMES, analyzer).parse("\"" + name + "\"~" + PARAM_PROXIMITY), 1.0f);
      //Query anchorQ   = new BoostQuery(new QueryParser(FIELD_NAMES, analyzer).parse("\"" + query + "\"~1"), 1.0f);
      //Query fuzzyQ    = new BoostQuery(new QueryParser(FIELD_NAMES, analyzer).parse(splitString(name, "~" + PARAM_FUZZY)), 0.1f);
      //Query redirectQ;
      BooleanQuery query = new BooleanQuery.Builder()
            .add(exactQ, BooleanClause.Occur.SHOULD)
            //.add(redirectQ, BooleanClause.Occur.SHOULD)
            //.add(fuzzyQ, BooleanClause.Occur.SHOULD)
            .build();
      
      TopDocs top = searcher.search(query, hits);
      //log.info(q.toString());
      ScoreDoc[] docs = top.scoreDocs;
      for(ScoreDoc hit : docs) {
        Document d = searcher.doc(hit.doc);
        result.add(d);
      }
      
    } catch(Exception ex) {
      log.error(ex.toString());
    }
    
    return result;
    
  }
  
  protected List<ArticleRef> queryIndexPrefix(String prefix, int hits) {
    
    List<ArticleRef> result = new ArrayList<>();
    
    try {
      Query exactQ    = new BoostQuery(new QueryParser(FIELD_NAMES, analyzer).parse("\"" + prefix + "\"*"), 1.0f);
      //Query anchorQ   = new BoostQuery(new QueryParser(FIELD_NAMES, analyzer).parse("\"" + query + "\"~1"), 1.0f);
      //Query fuzzyQ    = new BoostQuery(new QueryParser(FIELD_NAMES, analyzer).parse(splitString(name, "~" + PARAM_FUZZY)), 0.1f);
      //Query redirectQ;
      BooleanQuery query = new BooleanQuery.Builder()
            .add(exactQ, BooleanClause.Occur.SHOULD)
            //.add(redirectQ, BooleanClause.Occur.SHOULD)
            //.add(fuzzyQ, BooleanClause.Occur.SHOULD)
            .build();
      
      TopDocs top = searcher.search(query, hits);
      //log.info(q.toString());
      ScoreDoc[] docs = top.scoreDocs;
      for(ScoreDoc hit : docs) {
        Document d = searcher.doc(hit.doc);
        ArticleRef ref = createWikidataArticleRef(d);
        ref.setScore(hit.score);
        result.add(ref);
      }
      
    } catch(Exception ex) {
      log.error(ex.toString());
    }
    
    return result;
    
  }
  
  protected Optional<Document> queryIndexID(String field, String id) {
    try {
      Query query = new QueryParser(field, analyzer).parse("\"" + id + "\"");
      //log.info(query.toString());
      TopDocs top = searcher.search(query, 1);
      if(top.scoreDocs.length > 0) {
        ScoreDoc hit = top.scoreDocs[0];
        Document d = searcher.doc(hit.doc);
        return Optional.ofNullable(d);
      }
    } catch(Exception ex) {
      log.error(ex.toString());
    }
    return Optional.empty();
  }

}
