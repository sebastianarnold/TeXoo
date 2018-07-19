package de.datexis.models.index.impl;

import de.datexis.common.Resource;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import info.debatty.java.stringsimilarity.JaroWinkler;
import info.debatty.java.stringsimilarity.interfaces.NormalizedStringSimilarity;
import java.io.IOException;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class LuceneIndex {

  protected final static Logger log = LoggerFactory.getLogger(LuceneIndex.class);

  protected IndexReader reader;
  protected IndexSearcher searcher;
  protected Analyzer analyzer;
  protected final NormalizedStringSimilarity sim = new JaroWinkler();
  protected final TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
  
  /**
   * tries to open an existing index from given path
   */
  public boolean openIndex(Resource path) throws IOException {
    FSDirectory index = FSDirectory.open(path.getPath());
    return openIndex(index);
  }
  
  protected boolean openIndex(Directory index) throws IOException {
    reader = DirectoryReader.open(index);
    searcher = new IndexSearcher(reader);
    //searcher.setSimilarity(new BM25Similarity(1f,1f));
    //searcher.setSimilarity(new ContextSimilarity());
    analyzer = buildAnalyzer();
    return true;
  }
  
  protected abstract Analyzer buildAnalyzer();
  
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
  
  static final FieldType FIELDTYPE_NAME;
  static {
    FIELDTYPE_NAME = new FieldType();
    FIELDTYPE_NAME.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    FIELDTYPE_NAME.setTokenized(true);
    FIELDTYPE_NAME.setStored(true); // required for string similarity
    FIELDTYPE_NAME.setOmitNorms(true);
    FIELDTYPE_NAME.freeze();
  }
  
  
  
}
