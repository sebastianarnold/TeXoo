package de.datexis.cdv.reader;

import de.datexis.cdv.model.EntityAspectAnnotation;
import de.datexis.cdv.retrieval.EntityAspectQueryAnnotation;
import de.datexis.common.InternalResource;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Query;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.reader.DatasetReader;
import de.datexis.retrieval.model.RelevanceResult;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.xpath.*;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Reader for MedQuAD Dataset (https://github.com/abachaa/MedQuAD).
 * "A Question-Entailment Approach to Question Answering". Asma Ben Abacha and Dina Demner-Fushman. arXiv:1901.08079 [cs.CL], January 2019.
 *  @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MedQuADReader implements DatasetReader {
  
  protected final static Logger log = LoggerFactory.getLogger(MedQuADReader.class);
  
  protected XPath xpath;
  protected DocumentBuilder builder;
  
  private XPathExpression docIdQuery, docUrlQuery, docSourceQuery, docFocusQuery, docFocusIDsQuery, docFocusGroupQuery,
    docPassageQuery, docQuestionIdQuery, docQuestionAspectQuery, docQuestionTextQuery, docAnswerQuery;
  
  private Pattern subsetPattern = Pattern.compile(".+\\/(\\d+)_(.+)?\\/.+?\\.xml$");
  
  /** Map of all UMLS CUIS->URI and name->URI to map to a different ID scheme (e.g. Wikidata) */
  Map<String, String> umlsMap = null, namesMap = null, wikidataMap = null;
  
  protected boolean keepEmptyDocs = false, removeQuestions = false;
  protected Set<Class<? extends Annotation>> requestedAnnotations = new HashSet<>();
  
  protected List<String> labels;
  
  public MedQuADReader() {
    try {
      DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
      builder = factory.newDocumentBuilder();
      XPathFactory xPathfactory = XPathFactory.newInstance();
      xpath = xPathfactory.newXPath();
      docIdQuery = xpath.compile("/Document/@id");
      docUrlQuery = xpath.compile("/Document/@url");
      docSourceQuery = xpath.compile("/Document/@source");
      docFocusQuery = xpath.compile("/Document/Focus/text()");
      docFocusIDsQuery = xpath.compile("/Document/FocusAnnotations/UMLS/CUIs/CUI");
      docFocusGroupQuery = xpath.compile("/Document/FocusAnnotations/UMLS/SemanticGroup");
      docPassageQuery = xpath.compile("/Document/QAPairs/QAPair");
      docQuestionIdQuery = xpath.compile("./Question/@qid");
      docQuestionAspectQuery = xpath.compile("./Question/@qtype");
      docQuestionTextQuery = xpath.compile("./Question/text()");
      docAnswerQuery = xpath.compile("./Answer/text()");
    } catch(ParserConfigurationException | XPathExpressionException e) {
      e.printStackTrace();
    }
  }
  
  /**
   * Load the requested Annotation from MedQuAD (e.g. PassageAnnotation / EntityAspectAnnotation)
   */
  public MedQuADReader withAnnotations(Class<? extends Annotation> type) {
    this.requestedAnnotations.add(type);
    return this;
  }
  
  /**
   * Load a TSV file that contains mapping of UMLS CUI page to Wikidata IDs.
   */
  public MedQuADReader withIDMapping(Resource file) throws IOException {
    List<String> mapping = FileUtils.readLines(file.toFile(), "UTF-8");
    umlsMap = new ConcurrentHashMap<>(mapping.size());
    mapping.stream()
      .map(s -> s.split("\\t"))
      .forEach(s -> umlsMap.put(s[0], s[1]));
    return this;
  }
  
  /**
   * Load a TSV file that contains mapping of names to Wikidata IDs.
   */
  public MedQuADReader withNameMapping(Resource file) throws IOException {
    List<String> mapping = FileUtils.readLines(file.toFile(), "UTF-8");
    namesMap = new ConcurrentHashMap<>(mapping.size());
    wikidataMap = new ConcurrentHashMap<>(mapping.size());
    mapping.stream()
      .map(s -> s.split("\\t"))
      .forEach(s -> {
        namesMap.putIfAbsent(s[1].toLowerCase(), s[0]);
        wikidataMap.putIfAbsent(s[0], s[1]);
      });
    return this;
  }
  
  public MedQuADReader withPassageLabelsCSV(Resource csv) throws IOException {
    labels = IOUtils.readLines(csv.getInputStream(), StandardCharsets.UTF_8);
    return this;
  }
  
  /**
   * If TRUE, empty docs/abstracts are not discarded.
   */
  public MedQuADReader withKeepEmptyDocs(boolean keepEmptyDocs) {
    this.keepEmptyDocs = keepEmptyDocs;
    return this;
  }
  
  /**
   * If TRUE, questions at the beginning of passages will be removed
   */
  public MedQuADReader withRemoveQuestions(boolean removeQuestions) {
    this.removeQuestions = removeQuestions;
    return this;
  }
  
  @Override
  public Dataset read(Resource path) throws IOException {
    if(path instanceof InternalResource || path.isFile()) {
      Dataset data = new Dataset(path.getFileName());
      addDocumentFromFile(path, data, null);
      return data;
    } else if(path.isDirectory()) {
      return readDatasetFromDirectory(path, "\\.xml$");
    } else throw new FileNotFoundException("cannot open path: " + path.toString());
  }
  
  public Dataset readDatasetFromDirectory(Resource path, String pattern) throws IOException {
    log.info("Reading Documents from {}", path.toString());
    Dataset data = new Dataset(path.getPath().getFileName().toString());
    AtomicInteger progress = new AtomicInteger();
    ListIterator<String> labelIt = labels != null ? labels.listIterator() : Collections.emptyListIterator();
    if(labelIt.hasNext()) labelIt.next(); // skip header
    Files.walk(path.getPath())
      .filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))
      .filter(p -> p.toString().matches(pattern))
      .forEach(p -> {
        if(!addDocumentFromFile(Resource.fromFile(p.toString()), data, labelIt.hasNext() ? labelIt.next() : null)) {
          if(labelIt.hasPrevious()) labelIt.previous(); // revert document that was rejected
        }
        int n = progress.incrementAndGet();
        if(n % 1000 == 0) {
          double free = Runtime.getRuntime().freeMemory() / (1024. * 1024. * 1024.);
          double total = Runtime.getRuntime().totalMemory() / (1024. * 1024. * 1024.);
          log.debug("read {}k documents, memory usage {} GB", n / 1000, (int)((total-free) * 10) / 10.);
        }
      });
    return data;
  }
  
  public Dataset readDatasetFromFiles(Collection<String> files) throws IOException {
    Dataset data = new Dataset("MedQuAD");
    AtomicInteger progress = new AtomicInteger();
    ListIterator<String> labelIt = labels != null ? labels.listIterator() : Collections.emptyListIterator();
    if(labelIt.hasNext()) labelIt.next(); // skip header
    files.stream()
      .forEach(p -> {
        if(!addDocumentFromFile(Resource.fromFile(p), data, labelIt.hasNext() ? labelIt.next() : null)) {
          if(labelIt.hasPrevious()) labelIt.previous(); // revert document that was rejected
        }
        int n = progress.incrementAndGet();
        if(n % 1000 == 0) {
          double free = Runtime.getRuntime().freeMemory() / (1024. * 1024. * 1024.);
          double total = Runtime.getRuntime().totalMemory() / (1024. * 1024. * 1024.);
          log.debug("read {}k documents, memory usage {} GB", n / 1000, (int)((total-free) * 10) / 10.);
        }
      });
    return data;
  }
  
  protected boolean addDocumentFromFile(Resource path, Dataset data, String labels) {
  
    try {
      
      Document doc = new Document();
      
      org.w3c.dom.Document xml = builder.parse(path.getInputStream());
      Matcher setMatcher = subsetPattern.matcher(path.toString());
      if(!setMatcher.matches())
        throw new IllegalArgumentException("Invalid path structure. Please use the original MedQuAD folder.");
      String sourceId = (String) docSourceQuery.evaluate(xml, XPathConstants.STRING);
      doc.setId(sourceId + "_" + (String) docIdQuery.evaluate(xml, XPathConstants.STRING));
      //doc.setSource((String) xpath.compile("/Document/@source").evaluate(xml, XPathConstants.STRING));
      doc.setSource((String) docUrlQuery.evaluate(xml, XPathConstants.STRING));
      String focus = (String) docFocusQuery.evaluate(xml, XPathConstants.STRING);
      String group = (String) docFocusGroupQuery.evaluate(xml, XPathConstants.STRING);
      if(!group.equals("Disorders")) return false;
      doc.setTitle(focus);
      doc.setType(setMatcher.group(2).replaceFirst("_QA", ""));
      doc.setLanguage("en");
      // fix errors in dataset
      focus = focus.replace("What I need to know about ", "");
  
      // add given UMLS CUI to focus
      Set<String> umlsIDs = new TreeSet<>();
      String idString = null;
      NodeList list = (NodeList) docFocusIDsQuery.evaluate(xml, XPathConstants.NODESET);
      for(int j = 0; j < list.getLength(); j++) {
        Node node = list.item(j);
        umlsIDs.add(node.getTextContent());
      }
      List<String> ids = Collections.EMPTY_LIST;
      if(labels != null) {
        String[] label = labels.split("\\t");
        if(!label[0].equals(doc.getId()))
          log.error("got wrong label for docId {}", doc.getId());
        focus = label[1];
        idString = label.length > 2 ? label[2] : "";
      } else {
        //log.warn("no label given");
        if(umlsMap != null) {
          // match UMLS ids
          ids = umlsIDs.stream()
            .map(cui -> umlsMap.get(cui))
            .filter(Objects::nonNull)
            .distinct()
            .collect(Collectors.toList());
        }
        if(ids.isEmpty() && namesMap != null) {
          // match name
          String id = namesMap.get(focus.toLowerCase());
          if(id != null) ids.add(id);
        }
        if(umlsMap != null || namesMap != null) {
          idString = StringUtils.join(ids, ";");
          if(ids.isEmpty())
            log.warn("could not resolve ID for '{}' ({})", focus, StringUtils.join(umlsIDs, ";"));
        }
      }
      
      NodeList nl = (NodeList) docPassageQuery.evaluate(xml, XPathConstants.NODESET);
      for(int i = 0; i < nl.getLength(); i++) {
        Node qxml = nl.item(i);
        
        // append text
        String text = (String) docAnswerQuery.evaluate(qxml, XPathConstants.STRING) + "\n";
        if(removeQuestions) text = text.replaceFirst("^(.(?!\\. ))+?\\? ", ""); // remove questions at the beginning of paragraphs
        text = text.replace(" - ", "\n- "); // replace lists with newlines
        Document passage = DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);
        doc.append(passage); // no extra space needed
        
        // append question
        String passageId = sourceId + "_" + (String) docQuestionIdQuery.evaluate(qxml, XPathConstants.STRING);
        String queryText = DocumentFactory.createSentenceFromTokenizedString(
          (String) docQuestionTextQuery.evaluate(qxml, XPathConstants.STRING)).toString();
        
        // fix errors in dataset
        String aspect = (String) docQuestionAspectQuery.evaluate(qxml, XPathConstants.STRING);
        /*if(aspect.equals("information")) {
          aspect = null; // don't include generic "information" queries
         */
        if(sourceId.equals("NIHSeniorHealth") && aspect.equals("support groups") /*&& passageId.equals("7_0000002-29")*/) {
          aspect = "treatment"; // fix a single over-specific label
        } else if(sourceId.equals("GHR") && aspect.equals("treatment")) {
          aspect = null; // GHR treatments only contain generic links
        }
        
        // add Annotation for Question
        Query query = aspect != null ? Query.create(queryText) : null;
        //query.setDocumentRef(doc); // only do so if this query references a single Document
        //query.setId(qId);
        EntityAspectQueryAnnotation queryAnn = new EntityAspectQueryAnnotation(focus, aspect);
        
        if(idString != null) {
          queryAnn.setEntityId(idString);
        } else {
          queryAnn.setEntityId(StringUtils.join(umlsIDs, ";"));
        }
        
        // search if query already exists
        boolean merged = false;
        if(query != null) {
          for(Query other : data.getQueries()) {
            EntityAspectQueryAnnotation otherAnn = other.getAnnotation(EntityAspectQueryAnnotation.class);
            if(otherAnn.matches(queryAnn)) {
              query = other;
              merged = true;
              break;
            }
          }
          if(!merged) query.addAnnotation(queryAnn);
        }
        
        // add Annotation for Passage
        PassageAnnotation passageAnn;
        if(requestedAnnotations.contains(EntityAspectAnnotation.class)) {
          // add GOLD information as well
          passageAnn = new EntityAspectAnnotation(Annotation.Source.GOLD);
          ((EntityAspectAnnotation)passageAnn).setAspect(queryAnn.getAspect());
          ((EntityAspectAnnotation)passageAnn).setEntity(queryAnn.getEntity());
          ((EntityAspectAnnotation)passageAnn).setEntityId(queryAnn.getEntityId());
        } else {
          passageAnn = new PassageAnnotation(Annotation.Source.GOLD);
          if(aspect != null && !aspect.isEmpty()) passageAnn.setLabel(aspect.substring(0, 1).toUpperCase() + aspect.substring(1));
        }
        passageAnn.setId(passageId);
        passageAnn.setBegin(passage.getBegin());
        passageAnn.setEnd(passage.getEnd());
        if(passageAnn.getLength() > 1) doc.addAnnotation(passageAnn); // also annotate the GOLD passage in the document
        
        // add Annotation for Answer
        if(query != null) {
          RelevanceResult resultAnnotation = new RelevanceResult(Annotation.Source.GOLD, doc, passage.getBegin(), passage.getEnd());
          resultAnnotation.setRelevance(1);
          resultAnnotation.setId(passageId);
          resultAnnotation.setDocumentRef(doc);
          if(resultAnnotation.getLength() > 1 || keepEmptyDocs) query.addResult(resultAnnotation);
    
          if(!merged && query.getResults().size() > 0) data.addQuery(query);
        }
        
      }
      
      if(!doc.isEmpty() || keepEmptyDocs) {
        data.addDocument(doc);
        return true;
      } else {
        return false;
      }
      
    } catch(SAXException | XPathExpressionException e) {
      throw new IllegalArgumentException(e.toString());
    } catch(IOException ex) {
      // IOException is now allowed in Stream
      log.error(ex.toString());
      throw new RuntimeException(ex.toString(), ex.getCause());
    }
    
  }
  
  public void printQueries(Dataset corpus) {
    System.out.println("Query\tQuestion\tEntity\tQids\tResolved\tAspect");
    int i=0;
    for(Query query : corpus.getQueries()) {
      EntityAspectQueryAnnotation qann = query.getAnnotation(EntityAspectQueryAnnotation.class);
      StringBuilder out = new StringBuilder();
      out.append(i++).append("\t");
      out.append(query.getText()).append("\t");
      out.append(qann.getEntity()).append("\t");
      out.append(qann.getEntityId()).append("\t");
      if(qann.getEntityId() != null) for(String id : qann.getEntityId().split(";")) {
        out.append(wikidataMap.get(id)).append(";");
      }
      out.append("\t");
      out.append(qann.getAspect()).append("\t");
      System.out.println(out.toString());
    }
  }
  
  public void printDocuments(Dataset corpus) {
    System.out.println("DocId\tEntity\tQids\tResolved");
    int i=0;
    for(Document doc : corpus.getDocuments()) {
      EntityAspectAnnotation qann = doc.streamAnnotations(Annotation.Source.GOLD, EntityAspectAnnotation.class).findFirst().get();
      StringBuilder out = new StringBuilder();
      out.append(doc.getId()).append("\t");
      out.append(qann.getEntity()).append("\t");
      out.append(qann.getEntityId()).append("\t");
      if(qann.getEntityId() != null) for(String id : qann.getEntityId().split(";")) {
        out.append(wikidataMap.get(id)).append(";");
      }
      System.out.println(out.toString());
    }
  }
  
}
