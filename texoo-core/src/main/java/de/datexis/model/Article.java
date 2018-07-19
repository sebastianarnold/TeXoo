package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import java.util.Set;
import java.util.TreeSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Article is an (unparsed) Document from a Knowledge Base. Used for Entity Linking.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>, Denis Martin
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "class")
public class Article extends Span {

  protected final static Logger log = LoggerFactory.getLogger(Article.class);

  protected String id;
  protected String url;
  protected String title;
  protected String text;
  protected String description;
  protected String type;
  protected Set<String> names;
  protected Set<String> terms;
  protected INDArray vector;

  public Article(String id, String url, String title, String text) {
    terms = new TreeSet<>();
    names = new TreeSet<>();
    this.id = id;
    this.url = url;
    this.title = title;
    this.text = text;
    this.begin = 0;
    if(text != null) setLength(text.length());
    else setLength(0);
  }

  public Article() {
    terms = new TreeSet<>();
    names = new TreeSet<>();
  };

  public String getTitle() {
    return title;
  }

  @Override
  public String getText() {
    return text;
  }

 
  public void setTitle(String title) {
    this.title = title;
  }

  public void setText(String text) {
    this.text = text;
  }

  public String getId() {
    return id;
  }

  public void setId(String id) {
    this.id = id;
  }

  public String getUrl() {
    return url;
  }

  public void setUrl(String url) {
    this.url = url;
  }

  public String getDescription() {
    return description;
  }

  public void setDescription(String description) {
    this.description = description;
  }

  public String getType() {
    return type;
  }

  public void setType(String type) {
    this.type = type;
  }

  /**
   * Add a Term that references the Article.
   * @param term e.g. a Wikipedia anchor text.
   */
  public void addTerm(String term) {
    terms.add(term);
  }

  /**
   * Add a name to the Article. All names are treated equally.
   * @param name e.g. a Wikipedia redirect or alias
   */
  public void addName(String name) {
    names.add(name);
  }
  
  public void setNames(Set<String> names) {
    this.names = names;
  }
  
  public Set<String> getNames() {
    return names;
  }

  public void setTerms(Set<String> terms) {
    this.terms = terms;
  }

  public Set<String> getTerms() {
    return terms;
  }
  
  @JsonIgnore
  public INDArray getVector() {
    return vector;
  }

  public void setVector(INDArray vector) {
    this.vector = vector;
  }
  
}