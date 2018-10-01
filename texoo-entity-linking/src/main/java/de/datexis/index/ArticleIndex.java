package de.datexis.index;

import java.util.*;


/**
 * An Index holds a collection of Articles that are used as target for Entity Linking.
 * @author Denis Martin
 */
public abstract class ArticleIndex {

  /** used for JSON deserialization */
  public ArticleIndex() {}

  /**
   * Retrieve candidates for a query on the "name" field.
   * @param name The name to search, e.g. "obama barack"
   * @param hits Max number of hits to generate
   * @return An List of matching ArticleRefs, e.g. Wiki URL of "Barack Obama"
   */
  public abstract List<ArticleRef> queryNames(String name, int hits);
  
  /**
   * Retrieve candidates for auto completion on the "name" field.
   * @param prefix
   * @param hits
   * @return 
   */
  public abstract List<ArticleRef> queryPrefixNames(String prefix, int hits);
  
  /**
   * Retrieve the article with a given ID
   * @param id The indexed ID, e.g. "Q64"
   * @return The ArticleRef, if exists
   */
  public abstract Optional<ArticleRef> queryID(String id);
  
}
