package de.datexis.cdv.reader;

import de.datexis.cdv.model.AspectAnnotation;
import de.datexis.cdv.model.EntityAnnotation;
import de.datexis.cdv.retrieval.EntityAspectQueryAnnotation;
import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.retrieval.model.RelevanceResult;
import de.datexis.retrieval.preprocess.WikipediaUrlPreprocessor;
import de.datexis.sector.reader.WikiSectionReader;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class WikiSectionQAReader extends WikiSectionReader {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  /** Map of all UMLS URL->QID and name->URI to map to a different ID scheme (e.g. Wikidata) */
  Map<String, String> idMap = null;
  
  /**
   * Load a TSV file that contains mapping of UMLS CUI page to Wikidata IDs.
   */
  public WikiSectionQAReader withIDMapping(Resource file) throws IOException {
    List<String> mapping = FileUtils.readLines(file.toFile(), "UTF-8");
    idMap = new ConcurrentHashMap<>(mapping.size());
    mapping.stream()
      .map(s -> s.split("\\t"))
      .forEach(s -> idMap.put(WikipediaUrlPreprocessor.cleanWikiPageTitle(s[0]), s[1]));
    return this;
  }
  
  public Dataset convertTrainingPassages(Dataset corpus, boolean generateNegativeSamples) throws IOException {
    Dataset result = new Dataset();
    for(Document doc : corpus.getDocuments()) {
      String qid = idMap != null ? idMap.get(WikipediaUrlPreprocessor.cleanWikiPageTitle(doc.getId())) : doc.getId();
      // clean up annotations so that we have a stream of passages
      for(EntityAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, EntityAnnotation.class)) {
        doc.removeAnnotation(ann);
      }
      int count = 0;
      int cursor = 0;
      List<AspectAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, AspectAnnotation.class)
        .sorted()
        .collect(Collectors.toList());
      for(AspectAnnotation ann : anns) {
        // don't skip subsections
        //if(ann.getBegin() < cursor) continue;
        cursor = ann.getEnd();
        ann.setId(qid + "-" + count++);
        String label = ann.getLabel().replace(";", " ").toLowerCase();
        if(label.equals("abstract")) label = "information";
        Query query = Query.create(doc.getTitle() + " ; " + label);
        EntityAspectQueryAnnotation queryAnn = new EntityAspectQueryAnnotation(doc.getTitle(), label);
        queryAnn.setEntityId(qid);
        query.addAnnotation(queryAnn);
        RelevanceResult resultAnnotation = new RelevanceResult(Annotation.Source.GOLD, doc, ann.getBegin(), ann.getEnd());
        resultAnnotation.setRelevance(1);
        resultAnnotation.setId(ann.getId());
        resultAnnotation.setDocumentRef(doc);
        query.addResult(resultAnnotation);
        result.addQuery(query);
      }
      result.addDocument(doc);
    }
    if(generateNegativeSamples) {
      Random random = new Random();
      for(Query query : result.getQueries()) {
        // fill up with up to 10 random candidate passages
        Result matched = query.getResults().get(0);
        while(query.getResults().size() < 10) {
          de.datexis.model.Document doc = result.getRandomDocument().get();
          List<PassageAnnotation> anns = doc.streamAnnotations(Annotation.Source.GOLD, PassageAnnotation.class, true).collect(Collectors.toList());
          int idx = random.nextInt(anns.size());
          PassageAnnotation ann = anns.get(idx);
          if(ann.getId().equals(matched.getId())) continue; // already contained
          RelevanceResult resultAnnotation = new RelevanceResult(Annotation.Source.SAMPLED, doc, ann.getBegin(), ann.getEnd());
          resultAnnotation.setRelevance(0);
          resultAnnotation.setId(ann.getId());
          resultAnnotation.setDocumentRef(doc);
          query.addResult(resultAnnotation);
        }
      }
    }
    return result;
  }
  
}
