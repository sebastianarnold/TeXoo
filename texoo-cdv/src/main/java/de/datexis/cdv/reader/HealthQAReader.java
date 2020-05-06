package de.datexis.cdv.reader;

import de.datexis.cdv.model.EntityAspectAnnotation;
import de.datexis.cdv.retrieval.EntityAspectQueryAnnotation;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Query;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.retrieval.model.RelevanceResult;
import de.datexis.retrieval.preprocess.WikipediaUrlPreprocessor;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reader for HealthQA dataset (https://github.com/mingzhu0527/HAR)
 * "A Hierarchical Attention Retrieval Model for Healthcare Question Answering". Ming Zhu, Aman Ahuja, Wei Wei, Chandan Reddy. International Conference on World Wide Web (WWW) 2019.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class HealthQAReader extends MatchZooReader {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  /** Map of all UMLS CUIS->URI and name->URI to map to a different ID scheme (e.g. Wikidata) */
  Map<String, String> idMap = null, namesMap = null;
  
  Pattern entityPattern = Pattern.compile("^.+? (are|is|for|of|in|the|a|an)(?!.* (are|is|for|of|in|the|a|an) ) (.+) \\?$");
  
  protected Set<Class<? extends Annotation>> requestedAnnotations = new HashSet<>();
  
  protected List<String> labels;
  List<Map.Entry<String, String>> questions = new ArrayList<>();
  
  public HealthQAReader() {
    // some match strings to help for labeling
    questions.add(new AbstractMap.SimpleEntry<>("What is ", "information"));
    questions.add(new AbstractMap.SimpleEntry<>("What are ", "information"));
    questions.add(new AbstractMap.SimpleEntry<>(" infect", "infection"));
    questions.add(new AbstractMap.SimpleEntry<>(" common ", "prevalence"));
    questions.add(new AbstractMap.SimpleEntry<>(" cause", "causes"));
    questions.add(new AbstractMap.SimpleEntry<>(" symptom", "symptoms"));
    questions.add(new AbstractMap.SimpleEntry<>(" complication", "complications"));
    questions.add(new AbstractMap.SimpleEntry<>(" test", "diagnosis"));
    questions.add(new AbstractMap.SimpleEntry<>(" treat", "treatment"));
    questions.add(new AbstractMap.SimpleEntry<>(" outlook", "prognosis"));
    questions.add(new AbstractMap.SimpleEntry<>(" prognosis", "prognosis"));
    questions.add(new AbstractMap.SimpleEntry<>(" diagnose", "diagnosis"));
    questions.add(new AbstractMap.SimpleEntry<>(" vaccin", "prevention"));
    questions.add(new AbstractMap.SimpleEntry<>(" prevent", "prevention"));
  }
  
  /**
   * Load the requested Annotation from HealthQA (e.g. PassageAnnotation / EntityAspectAnnotation)
   */
  public HealthQAReader withAnnotations(Class<? extends Annotation> type) {
    this.requestedAnnotations.add(type);
    return this;
  }
  
  /**
   * Load a TSV file that contains mapping of names to Wikidata IDs.
   */
  public HealthQAReader withNameMapping(Resource file) throws IOException {
    List<String> mapping = FileUtils.readLines(file.toFile(), "UTF-8");
    namesMap = new ConcurrentHashMap<>(mapping.size());
    mapping.stream()
      .map(s -> s.split("\\t"))
      .forEach(s -> namesMap.putIfAbsent(s[1].toLowerCase(), s[0]));
    return this;
  }
  
  /**
   * Load a TSV file that contains mapping of Wikidata IDs to Wikipedia Page Names.
   */
  public HealthQAReader withIDMapping(Resource file) throws IOException {
    List<String> mapping = FileUtils.readLines(file.toFile(), "UTF-8");
    idMap = new ConcurrentHashMap<>(mapping.size());
    mapping.stream()
      .map(s -> s.split("\\t"))
      .forEach(s -> idMap.put(s[1], WikipediaUrlPreprocessor.cleanWikiPageTitle(s[0])));
    return this;
  }
  
  public HealthQAReader withPassageLabelsCSV(Resource csv) throws IOException {
    labels = IOUtils.readLines(csv.getInputStream(), StandardCharsets.UTF_8);
    return this;
  }
  
  public void resolveEntityLabels() {
    for(String row : labels) {
      String[] col = row.split("\t");
      List<String> ids = new ArrayList<>();
      String pageName = "";
      if(col.length > 3) {
        String entities = col[3];
        for(String entity : entities.split(";")) {
          if(namesMap != null) {
            // match name
            String id = namesMap.get(entity.toLowerCase());
            if(id != null) ids.add(id);
          }
          if(idMap != null && !ids.isEmpty()) {
            // get page titles
            String name = idMap.get(ids.get(0));
            if(name != null) pageName = name;
          }
        }
      }
      System.out.println(col[0] + "\t" + String.join(";", ids) + "\t" + pageName.replace("_", " "));
    }
  }
  
  protected void addDocumentFromFile(Resource path, Dataset data) throws IOException {
    
    data.setName("HealthQA");
    data.setLanguage("en");
    
    String fileName = path.getFileName().replaceAll("\\.txt$","");
    
    try(InputStream in = path.getInputStream()) {
      CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
      Iterator<String> lineIt = new LineIterator(br);
      Iterator<String> labelIt = labels != null ? labels.iterator() : null;
      if(labelIt != null) labelIt.next(); // skip title
      String[] line, label;
  
      Document doc = new Document();
      int lineIndex = -1;
      
      while(lineIt.hasNext()) {
      
        // read next passage
        line = lineIt.next().split("\\t");
        lineIndex++;
        boolean relevant = line[0].trim().equals("1");
        /*if(relevant) {
          // print out questions for labeling
          matchPassage(lineIndex, line[1]);
          continue;
        }*/
        if(!relevant) continue; // skip negative examples
        label = labelIt.next().split("\\t");
        String passageId = fileName + "-" + lineIndex;
        boolean docStart = label[1].equals("1");
        String question = line[1];
        assert question.equals(label[2]);
        String text = line[2] + "\n";
        String entityId = label[3];
        String entity = label[4];
        String type = label[5];
        String aspect = label[6];
      
        if(docStart) {
          // add last document
          if(!doc.isEmpty()) data.addDocument(doc);
          // start new document
          doc = new Document();
          doc.setId(fileName + "-doc-" + lineIndex);
          doc.setTitle(entity);
        }
  
        // append text
        if(text.substring(text.length() - 2, text.length() - 1).equals(","))
          text = text.substring(0, text.length() - 2);
        text = DocumentFactory.fromTokenizedText(text).getText() + "\n"; // retokenize the clean text
        Document passage = DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);
        doc.append(passage); // TODO: do we need an extra space?
  
        // append query
        Query query = Query.create(question);
        EntityAspectQueryAnnotation queryAnn = new EntityAspectQueryAnnotation(entity, aspect);
        queryAnn.setEntityId(entityId);
  
        // search if query already exists
        boolean merged = false;
        for(Query other : data.getQueries()) {
          EntityAspectQueryAnnotation otherAnn = other.getAnnotation(EntityAspectQueryAnnotation.class);
          if(otherAnn.matches(queryAnn)) {
            query = other;
            merged = true;
            break;
          }
        }
        if(!merged) query.addAnnotation(queryAnn);
  
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
          passageAnn.setLabel(question);
        }
        passageAnn.setId(passageId);
        passageAnn.setBegin(passage.getBegin());
        passageAnn.setEnd(passage.getEnd());
        if(passageAnn.getLength() > 1) doc.addAnnotation(passageAnn); // also annotate the GOLD passage in the document
  
        // add Annotation for Answer
        RelevanceResult resultAnnotation = new RelevanceResult(Annotation.Source.GOLD, doc, passage.getBegin(), passage.getEnd());
        resultAnnotation.setRelevance(1);
        resultAnnotation.setId(passageId);
        resultAnnotation.setDocumentRef(doc);
        query.addResult(resultAnnotation);
  
        if(!merged && query.getResults().size() > 0) data.addQuery(query);
      
      }
  
      // add last document
      if(!doc.isEmpty()) data.addDocument(doc);
      
      assert !labelIt.hasNext();
      
    }
  }
  
  protected void matchPassage(int lineNum, String question) {
    Matcher entityMatcher = entityPattern.matcher(question);
    String entity = entityMatcher.matches() ? entityMatcher.group(3) : "";
    String aspect = "";
    for(Map.Entry<String,String> match : questions) {
      if(question.contains(match.getKey())) aspect = match.getValue();
    }
    System.out.println(lineNum + "\t" + question + "\t" + entity + "\t" + aspect);
  }
  
}
