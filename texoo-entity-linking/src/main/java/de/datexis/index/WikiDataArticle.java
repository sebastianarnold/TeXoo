package de.datexis.index;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.datexis.nel.model.Article;

import java.util.*;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * WikiDataArticle JSON Structure
 * {
 *  "id" : "Q31",
 *  "title" : "Belgium", // Label in der jeweiligen Sprache [DE, EN]
 *  "type" : "State",
 *  "refIDs" : [
 *      { "wikidata" : "Q31" },
 *      { "freebase" : "/m/0154j" },
 *      { "ICD-10" : null },
 *      { "UMLS" : null },
 *      { "wikipedia" : "800" },
 *  ],
 *  "refURLs" : [
 *      { "image" : "https://commons.wikimedia.org/wiki/File:Europe_location_BEL.png" },
 *      { "wikipedia" : "http://en.wikipedia.org/wiki/Belgium" },
 *      { "dbpedia" : null },
 *      { "springer" : null },
 *  ]
 *  "text" : "blah blah Abstract (~3 sentences)",
 *  "description" : "constitutional monarchy in Western Europe (~2 paragraphs)",
 *  "names" : [ "Belgium", "Kingdom_of_Belgium", "be" ], // alle Labels und Aliases in EN und DE
 *  "terms" : [ ],
 *  // ab hier noch unklar
 *  "vectors" : { "de.datexis.encoder.impl.DocumentEncoder" : "AAdKQVZBQ1BQAAAACAADSU5UAAAAAgAAAAEAAACWAAAAAQAAAAEAAAAAAAAAAQAAAGMAB0pBVkFD" },
 *  "scorePopularity" : 1246,
 * }
 */

@JsonIgnoreProperties(ignoreUnknown = true)
public class WikiDataArticle extends Article {

  public Map<String,String> refIDs; // we use <String,String> and not <RefID,String> for easy JSON serialization across versions
  public Map<String,String> refURLs;
  public Set<String> types;

  public static enum RefID { WIKIDATA, FREEBASE, WIKIPEDIA, UMLS, ICD10 };

  public WikiDataArticle() {
    this.refIDs = new TreeMap<>();
    this.refURLs = new TreeMap<>();
    this.types = new TreeSet<>();
  }

  public WikiDataArticle(String id, String title, String type, Map<String, String> refIDs, Map<String, String> refURLs, String description, Set<String> names, Set<String> terms) {
    this.id = id;
    this.title = title;
    this.type = type;
    this.refIDs = new TreeMap<>(refIDs);
    this.refURLs = new TreeMap<>(refURLs);
    this.types = new TreeSet<>();
    this.description = description;
    this.names = names;
    this.terms = terms;
  }

  public void setTypes(Set<String> types) {
    this.types = types;
  }
  
  /**
   * @return highest level of assigned types and subcclasses. Sub-types are not resolved yet!
   */
  public Set<String> getTypes() {
    return types;
  }
  
  public void addType(String type) {
    types.add(type);
  }
  
  public Map<String,String> getRefIDs() {
    return refIDs;
  }

  @JsonIgnore
  public Map<String,String> getRefURLs() {
    return refURLs;
  }

  public void putRefID(RefID key, String value) {
    if(value != null) refIDs.put(key.toString().toLowerCase(), value);
  }
  
  @JsonIgnore
  public String getRefID(RefID key) {
    return refIDs.getOrDefault(key.toString().toLowerCase(), null);
  }

  @JsonIgnore
  public String getRefURL(String key) {
    return refURLs.getOrDefault(key, null);
  }

  /*@Override
  public String getUrl() {
    return refIDs.getOrDefault(RefID.WIKIPEDIA.toString().toLowerCase(), null);
  }*/
  
  @Override
  @JsonIgnore
  public String getType() {
    return super.getType();
  }

  @Override
  @JsonIgnore
  public int getBegin() {
    return super.getBegin();
  }

  @Override
  @JsonIgnore
  public int getLength() {
    return super.getLength();
  }

  @Override
  @JsonIgnore
  public String getText() {
    return super.getText();
  }
  
  @Override
  public String toString() {
    if(getTitle() != null && getId().equals("NIL")) return "NIL";
    else return getTitle() + " (" + getId() + ")";
  }

}
