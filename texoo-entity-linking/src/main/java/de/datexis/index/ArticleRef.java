package de.datexis.index;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import de.datexis.model.Article;
import java.util.Comparator;
import java.util.Objects;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class only holds a reference to an Article in the Knowledge Base, not the Text.
 * Used for Candidate generation.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "class")
public class ArticleRef {

  protected final static Logger log = LoggerFactory.getLogger(ArticleRef.class);

  protected String id;
  protected String url;
  protected String title;
  protected String description;
  protected String type;
  protected double score = 0.;
  protected INDArray vector;
  
  public ArticleRef() {
  }
  
  /**
   * Copy Constructor
   * @param article 
   */
  public ArticleRef(Article article) {
    this.id = article.getId();
    this.url = article.getUrl();
    this.title = article.getTitle();
    this.description = article.getDescription();
    this.type = article.getType();
    this.vector = article.getVector();
  }

  @JsonIgnore
  public INDArray getVector() {
    return vector;
  }

  public void setVector(INDArray vector) {
    this.vector = vector;
  }
  
  public double getScore() {
    return score;
  }

  public void setScore(double score) {
    this.score = score;
  }
  
  public String getTitle() {
    return title;
  }

  public String getId() {
    return id;
  }
  
  public String getUrl() {
    return url;
  }
  
  public String getDescription() {
    return description;
  }
  
  public String getType() {
    return type;
  }

  public void setId(String id) {
    this.id = id;
  }

  public void setUrl(String url) {
    this.url = url;
  }

  public void setTitle(String title) {
    this.title = title;
  }

  public void setDescription(String description) {
    this.description = description;
  }

  public void setType(String type) {
    this.type = type;
  }
  
  @Override
  public String toString() {
    if(getTitle() != null && getId().equals("NIL")) return "NIL";
    else return getTitle() + " (" + getId() + ")";
  }
  
  public static class ScoreComparator implements Comparator<ArticleRef> {
    @Override
    public int compare(ArticleRef o1, ArticleRef o2) {
      if(Objects.equals(o1.id, o2.id)) return 0;
      return Double.compare(o2.score, o1.score);
    }
  }
  
}
