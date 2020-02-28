package de.datexis.nel.index;

import com.google.common.collect.Lists;
import de.datexis.encoder.impl.TrigramEncoder;
import de.datexis.nel.model.NamedEntity;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.stream.Collectors;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class KnowledgeBaseTest {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  @Test
  public void testMentionRetrieval() {
    
    TrigramEncoder tri = new TrigramEncoder();
    tri.trainModel(Lists.newArrayList(DocumentFactory.fromText("Paracetamol, also known as acetaminophen and APAP, is a medication used to treat pain and fever.")));
    KnowledgeBase kb = new KnowledgeBase(tri, tri);
    
    NamedEntity e1 = new NamedEntity();
    e1.setId("Q57055");
    e1.setName("acetaminophen");
    e1.addLink("wikidata", "Q57055");
    e1.addLink("wikipedia", "Paracetamol");
    e1.addLink("umls", "C0000970");
    e1.setAliases(Lists.newArrayList("acetaminophen", "Paracetamol", "Acetamol", "Paracetanol"));
    e1.setDescription("common drug for pain and fever");
    
    NamedEntity e2 = new NamedEntity();
    e2.setId("Q18216");
    e2.setName("aspirin");
    e2.addLink("wikidata", "Q18216");
    e2.addLink("wikipedia", "Aspirin");
    e2.addLink("umls", "C0004057");
    e2.setAliases(Lists.newArrayList("aspirin", "Aspirin", "ASA", "2-Acetoxybenzoic acid"));
    e2.setDescription("medication used to treat pain and decrease the risk of heart disease");
  
    kb.addEntity(e1);
    kb.addEntity(e2);
    kb.buildIndex();
    
    assertThat(kb.countEntities(), is(2));
    assertThat(kb.countMentions(), is(8));
  
    assertThat(kb.getEntity("Q57055"), equalTo(e1));
    assertThat(kb.getEntity("Q18216"), equalTo(e2));
  
    List<KnowledgeBase.EntityCandidate> candidates = kb.findNearestEntities("Aspirin", 10);
    System.out.println(candidates);
    assertThat(candidates.size(), greaterThan(0));
    List<NamedEntity> result = candidates.stream().map(c -> c.entity).collect(Collectors.toList());
    assertThat(result, containsInRelativeOrder(e2));
    assertThat(candidates.get(0).similarity, greaterThan(0.99));
    assertThat(result, not(containsInRelativeOrder(e1)));
  
    candidates = kb.findNearestEntities("Paracetamol", 10);
    System.out.println(candidates);
    assertThat(candidates.size(), greaterThan(0));
    result = candidates.stream().map(c -> c.entity).collect(Collectors.toList());
    assertThat(candidates.get(0).similarity, greaterThan(0.99));
    assertThat(result, containsInRelativeOrder(e1));
  
    candidates = kb.findNearestEntities("ace", 10);
    System.out.println(candidates);
    assertThat(candidates.size(), greaterThan(0));
    result = candidates.stream().map(c -> c.entity).collect(Collectors.toList());
    assertThat(result, containsInRelativeOrder(e1, e2));
  
    candidates = kb.findNearestEntities("123456", 10);
    assertThat(candidates, empty());
    
  }
  
}
