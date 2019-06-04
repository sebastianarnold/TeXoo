package de.datexis.retrieval.preprocess;

import org.apache.commons.lang.StringEscapeUtils;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class WikipediaUrlPreprocessor implements TokenPreProcess {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  public WikipediaUrlPreprocessor() {
  }
  
  @Override
  public String preProcess(String pageId) {
    return cleanWikiPageTitle(pageId);
  }
  
  /**
   * Clean a given URL or page title to get a valid Wiki page title
   * see https://en.wikipedia.org/wiki/Wikipedia:Page_name#Technical_restrictions_and_limitations
   */
  public static String cleanWikiPageTitle(String pageTitle) {
    // strip URL prefix
    pageTitle = pageTitle.replaceFirst("^.+\\/wiki\\/", "").replaceFirst("#.+$", ""); // remove host path and anchors
    pageTitle = pageTitle.replaceAll("%(?![0-9A-F][0-9A-F])","%25"); // replace all '%' without following hex number by %25
    try {
      pageTitle = URLDecoder.decode(pageTitle, "UTF-8"); // replace escaped chars
    } catch(UnsupportedEncodingException | IllegalArgumentException e) {
    }
    
    pageTitle = StringEscapeUtils.unescapeHtml(pageTitle);
    return pageTitle.replace(" ","_").trim(); // replace spaces
  }
  
}
