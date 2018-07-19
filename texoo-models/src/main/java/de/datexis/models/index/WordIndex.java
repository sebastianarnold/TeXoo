package de.datexis.models.index;

import java.util.*;


/**
 * A simple Fulltext Index.
 */
public interface WordIndex {

  /**
   * Retrieve candidates for a Proximity query on the "text" field.
   */
  public List<String> queryText(String text, int hits);
  
  /**
   * Retrieve candidates for an exact  query on the "text" field.
   */
  public List<String> queryExactText(String text, int hits);
  
  /**
   * Retrieve candidates for auto completion on the "text" field.
   */
  public List<String> queryPrefixText(String prefix, int hits);
  
}
