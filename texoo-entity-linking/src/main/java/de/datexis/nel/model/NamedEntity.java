package de.datexis.nel.model;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * Knowledge Base entry for a named entity.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class NamedEntity {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected String id;
  protected String name;
  protected Collection<String> aliases;
  protected String description;
  protected Multimap<String, String> links = TreeMultimap.create();
  
  /**
   * @return the unique identifier of this entity (e.g. Wikidata ID)
   */
  public String getId() {
    return id;
  }
  
  public void setId(String id) {
    this.id = id;
  }
  
  /**
   * @return the canonical name for this entity
   */
  public String getName() {
    return name;
  }
  
  public void setName(String name) {
    this.name = name;
  }
  
  /**
   * @return a collection of alias names for this entity. Please make sure to include the canonical name.
   */
  public Collection<String> getAliases() {
    return aliases;
  }
  
  public void setAliases(Collection<String> aliases) {
    this.aliases = aliases;
  }
  
  /**
   * @return a description of this entity, can be used for training
   */
  public String getDescription() {
    return description;
  }
  
  public void setDescription(String description) {
    this.description = description;
  }
  
  /**
   * @return a multimap for links to foreign keys, e.g. "wikipedia" -> url, "UMLS" -> CUI
   */
  public Multimap<String, String> getLinks() {
    return links;
  }
  
  public void setLinks(Multimap<String, String> links) {
    this.links = links;
  }
  
  public Collection<String> getLinks(String key) {
    return links.get(key);
  }
  
  public void addLink(String key, String value) {
    links.put(key, value);
  }
  
}
