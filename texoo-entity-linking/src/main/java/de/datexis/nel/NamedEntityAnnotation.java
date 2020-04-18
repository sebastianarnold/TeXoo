package de.datexis.nel;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.google.common.base.Objects;
import com.google.common.collect.Lists;
import de.datexis.index.ArticleRef;
import de.datexis.model.Annotation;
import de.datexis.nel.model.NamedEntity;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

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
  protected NamedEntity entity;
  protected List<NamedEntity> candidates = new ArrayList<>();

  public NamedEntityAnnotation() {
  };

  public NamedEntityAnnotation(Source source, String text, int begin, int length, String type, String annotationSource) {
    super(source, text, begin, begin + length);
    this.type = type;
  }
 
  public NamedEntityAnnotation(de.datexis.ner.MentionAnnotation ann, List<ArticleRef> candidates) {
    super(ann);
    type = ann.getType();
    confidence = ann.getConfidence();
    setCandidates(candidates, true);
  }
  
  public NamedEntityAnnotation(NamedEntityAnnotation ann, List<ArticleRef> candidates) {
    super(ann);
    refName = ann.refName;
    refId = ann.refId;
    type = ann.type;
    confidence = ann.getConfidence();
    setCandidates(candidates, true);
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
    this.candidates = candidates.stream()
      .map(ref -> {
        NamedEntity c = new NamedEntity();
        c.setId(ref.getId());
        c.setAliases(Lists.newArrayList(ref.getTitle()));
        c.setName(ref.getTitle());
        c.addLink("url", ref.getUrl());
        return c;
      })
      .collect(Collectors.toList());
    if(setId) {
      if(this.candidates != null && !this.candidates.isEmpty()) {
        this.refId = this.candidates.get(0).getId();
        this.refName = this.candidates.get(0).getName();
        this.refUrl = this.candidates.get(0).getLinks("url").stream().findFirst().orElse(null);
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
  
  public void setEntity(NamedEntity entity) {
    this.entity = entity;
  }
  
  public NamedEntity getEntity() {
    return entity;
  }
  
  public List<NamedEntity> getCandidates() {
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
