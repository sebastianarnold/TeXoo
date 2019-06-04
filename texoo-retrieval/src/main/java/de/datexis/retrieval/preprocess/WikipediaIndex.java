package de.datexis.retrieval.preprocess;

import de.datexis.common.Resource;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.QuoteMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class WikipediaIndex {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected final String PAGE_LINE = "INSERT INTO `page` VALUES ";
  protected final String REDIRECT_LINE = "INSERT INTO `redirect` VALUES ";
  
  /** Map of all Wikipedia pages ID->Title (no redirects) */
  protected Map<Long,String> pageTitles = new ConcurrentHashMap<>(5000000);
  
  /** Map of all Wikipedia pages Title->ID (including redirects) */
  protected Map<String,Long> pageIndex = new ConcurrentHashMap<>(15000000);
  
  /** Map of all Redirects ID-Title (non-transitive) */
  protected Map<Long,String> pageRedirects = new ConcurrentHashMap<>(15000000);
  
  protected long matched = 0;
  protected long unmatched = 0;
  
  // TODO: make this Builder.build()
  public void readPages(Resource pageSql) throws IOException {
  
    CSVFormat csv = CSVFormat.DEFAULT
      .withAllowMissingColumnNames()
      .withDelimiter(',')
      .withRecordSeparator('\n')
      .withQuote('\'')
      .withQuoteMode(QuoteMode.NON_NUMERIC)
      .withEscape('\\')
      .withNullString("NULL");
  
    log.info("Reading Wikipedia pages from {}...", pageSql.toString());
    AtomicLong count = new AtomicLong();
    try(InputStream in = pageSql.getInputStream()) {
      CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
      br.lines()
        .parallel()
        .forEach(line -> {
          if(!line.startsWith(PAGE_LINE)) return;
          String batch = line.substring(PAGE_LINE.length() + 1, line.length() - 2).replace("),(", "\n");
          try(CSVParser parser = csv.parse(new StringReader(batch))) {
            for(CSVRecord row : parser.getRecords()) {
              long id = Long.parseLong(row.get(0));
              int namespace = Integer.parseInt(row.get(1));
              String title = row.get(2);
              boolean isRedirect = row.get(5).equals("1");
              if(namespace == 0) {
                if(title == null) {
                  if(id == 81447) title = "NULL"; // CSV reader seems to read 'NULL" (quoted null) as null
                  else log.warn("title is null: {}", row.toString());
                }
                if(!isRedirect && !title.endsWith("(disambiguation)"))
                  pageTitles.putIfAbsent(id, title);
                pageIndex.putIfAbsent(title, id);
                long n = count.incrementAndGet();
                if(n % 1000000 == 0) {
                  double free = Runtime.getRuntime().freeMemory() / (1024. * 1024. * 1024.);
                  double total = Runtime.getRuntime().totalMemory() / (1024. * 1024. * 1024.);
                  log.debug("read {}M rows, memory usage {} GB", n / 1000000, (int) ((total - free) * 10) / 10.);
                }
              }
            }
          } catch(IOException ex) {
            log.error(ex.toString());
          }
        });
    }
    log.info("Read {} entities out of total {} pages", pageTitles.size(), pageIndex.size());
  }
  
  public void readRedirects(Resource redirectSql) throws IOException {
    
    CSVFormat csv = CSVFormat.DEFAULT
      .withAllowMissingColumnNames()
      .withDelimiter(',')
      .withRecordSeparator('\n')
      .withQuote('\'')
      .withEscape('\\')
      .withNullString("NULL");
  
    log.info("Reading Wikipedia redirects from {}...", redirectSql.toString());
    AtomicLong count = new AtomicLong();
    try(InputStream in = redirectSql.getInputStream()) {
      CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
      br.lines()
        .parallel()
        .forEach(line -> {
          if(!line.startsWith(REDIRECT_LINE)) return;
          String batch = line.substring(REDIRECT_LINE.length() + 1, line.length() - 2).replace("),(", "\n");
          try(CSVParser parser = csv.parse(new StringReader(batch))) {
            for(CSVRecord row : parser.getRecords()) {
              long sourceId = Long.parseLong(row.get(0));
              int namespace = Integer.parseInt(row.get(1));
              String targetTitle = row.get(2);
              if(namespace == 0) {
                pageRedirects.putIfAbsent(sourceId, targetTitle);
                long n = count.incrementAndGet();
                if(n % 1000000 == 0) {
                  double free = Runtime.getRuntime().freeMemory() / (1024. * 1024. * 1024.);
                  double total = Runtime.getRuntime().totalMemory() / (1024. * 1024. * 1024.);
                  log.debug("read {}M rows, memory usage {} GB", n / 1000000, (int) ((total - free) * 10) / 10.);
                }
              }
            }
          } catch(IOException ex) {
            log.error(ex.toString());
          }
        });
    }
    log.info("Read {} redirects", pageRedirects.size());
  }
  
  public long countPages() {
    return pageTitles.size();
  }
  
  /**
   * @return pageTitle from Wikipedia page ID (no redirects)
   */
  public String getTitleForId(long pageId) {
    return pageTitles.get(pageId);
  }
  
  /**
   * @return redirected pageTitle from any Wikipedia pageTitle
   */
  public String getTitleFromRedirect(String pageTitle) {
    Long id = getIdForTitle(pageTitle);
    if(id == null) return null;
    else return getTitleForId(id);
  }
  
  /**
   * @return redirected pageId from any Wikipedia pageTitle (including redirects)
   */
  public Long getIdForTitle(String requestedPage) {
    if(requestedPage == null) return null;
    String redirectedPage = requestedPage;
    int redirects = 0;
    do {
      Long id = pageIndex.get(redirectedPage);
      // retry with first char uppercase
      if(id == null) id = redirectedPage.length() > 0 ? pageIndex.get(redirectedPage.substring(0, 1).toUpperCase() + (redirectedPage.length() > 1 ? redirectedPage.substring(1) : "")) : null;
      if(id == null) {
        // not matched
        unmatched++;
        //log.debug("Page '{}' not found ({} redirects)", redirectedPage, redirects);
        return null;
      } else if(pageRedirects.containsKey(id)) {
        // redirected
        redirectedPage = pageRedirects.get(id);
        if(redirectedPage != null && redirectedPage.equals(requestedPage)) {
          // cyclic redirect
          matched++;
          return id;
        }
      } else {
        // matched
        matched++;
        return id;
      }
    } while(++redirects < 32);
    log.error("Page id not found for '{}' after {} redirects", redirectedPage, redirects);
    return null;
  }
  
  public String getStats() {
    return "WikipediaIndex: " + matched + " matched, " + unmatched + " unmatched.";
  }
  
  // TODO: create an index here that contains the pages and all their redirects
  public void filterPages(List<String> pages) {
    Map<Long,String> prunedTitles = new HashMap<>(pages.size());
    for(String page : pages) {
      // strip URL
      page = WikipediaUrlPreprocessor.cleanWikiPageTitle(page);
      Long id = pageIndex.get(page); // check if page exists
      if(id == null) {
        log.info("Page '{}' not found in index", page);
      } else if(pageRedirects.containsKey(id)) {
        String redirect = getTitleFromRedirect(page);
        id = pageIndex.get(redirect);
        log.info("Page '{}' is a redirect to {}", page, redirect);
        prunedTitles.putIfAbsent(id, redirect);
      } else {
        prunedTitles.putIfAbsent(id, page);
      }
    }
    pageTitles = prunedTitles;
  }
  
}
