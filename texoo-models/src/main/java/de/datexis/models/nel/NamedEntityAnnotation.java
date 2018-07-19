package de.datexis.models.nel;

import de.datexis.model.Annotation;
import de.datexis.models.index.ArticleRef;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.google.common.base.Objects;

import java.util.ArrayList;
import java.util.List;

/**
 * Annotation of a named entity with candidates.
 * @author sarnold
 */

@JsonIgnoreProperties(ignoreUnknown = true)
public class NamedEntityAnnotation extends Annotation {
  
  protected String type;
  protected String refName;
  protected String refId;
  protected String refUrl;
  protected List<ArticleRef> candidates = new ArrayList<>();

  public NamedEntityAnnotation() {
  };

  public NamedEntityAnnotation(Source source, String text, int begin, int length, String type, String annotationSource) {
    super(source, text, begin, begin + length);
    this.type = type;
  }
 
  public NamedEntityAnnotation(de.datexis.models.ner.MentionAnnotation ann, List<ArticleRef> candidates) {
    super(ann);
    type = ann.getType();
    confidence = ann.getConfidence();
    this.candidates = candidates;
    if(candidates != null && !candidates.isEmpty()) {
      this.refName = candidates.get(0).getTitle();
      this.refId = candidates.get(0).getId();
      this.refUrl = candidates.get(0).getUrl();
    } else {
      this.refId = "NIL";
    }
  }
  
  public NamedEntityAnnotation(NamedEntityAnnotation ann, List<ArticleRef> candidates) {
    super(ann);
    refName = ann.refName;
    refId = ann.refId;
    type = ann.type;
    confidence = ann.getConfidence();
    this.candidates = candidates;
    if(candidates != null && !candidates.isEmpty()) {
      this.refName = candidates.get(0).getTitle();
      this.refId = candidates.get(0).getId();
      this.refUrl = candidates.get(0).getUrl();
    } else {
      this.refId = "NIL";
    }
  }

  public String getRefName() {
    return refName;
  }

  public void setRefName(String refName) {
    this.refName = refName;
  }

  public String getRefId() {
    return refId;
  }

  public void setRefId(String id) {
    this.refId = id;
  }

  public String getRefUrl() {
    return refUrl;
  }
  
  public void setRefUrl(String url) {
    this.refUrl = url;
  }
  
  public void setCandidates(List<ArticleRef> candidates, boolean setId) {
    this.candidates = candidates;
    if(setId) {
      if(candidates != null && !candidates.isEmpty()) {
        this.refName = candidates.get(0).getTitle();
        this.refId = candidates.get(0).getId();
        this.refUrl = candidates.get(0).getUrl();
      } else {
        this.refId = "NIL";
      }
    }
  }

  @Override
  public String toString() {
    if(getRefId() != null && getRefId().equals("NIL")) return "NIL";
    else return getRefName() + " (" + getRefId() + ")";
  }
  
  public String getType() {
    return type;
  }

  public List<ArticleRef> getCandidates() {
    return candidates;
  }
  
  @Override
  public boolean matches(Annotation other, Match match) {
    if(other instanceof NamedEntityAnnotation) {
      NamedEntityAnnotation that = (NamedEntityAnnotation) other;
      return super.matches(that, match) && (
        (this.getRefId()  != null && that.getRefId()  != null && Objects.equal(this.getRefId(),  that.getRefId())) || 
        (this.getRefUrl() != null && that.getRefUrl() != null && Objects.equal(this.getRefUrl(), that.getRefUrl()))
      );
    } else return false;
  }

}
