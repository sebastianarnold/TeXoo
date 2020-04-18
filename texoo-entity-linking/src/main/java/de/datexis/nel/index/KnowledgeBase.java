package de.datexis.nel.index;

import de.datexis.encoder.IEncoder;
import de.datexis.nel.model.NamedEntity;
import de.datexis.retrieval.index.IVectorIndex;
import de.datexis.retrieval.index.InMemoryIndex;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class KnowledgeBase {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  /** Encoder for entity mentions (names and aliases) */
  protected IEncoder mentionEncoder;
  
  /** Encoder for entity context (descriptions, articles, ...) */
  protected IEncoder contextEncoder;
  
  /** Index to resolve KNN vec(mention) -> ID */
  protected InMemoryIndex mentionIndex;
  
  protected Map<String, NamedEntity> entities;
  
  public KnowledgeBase(IEncoder mentionEncoder, IEncoder contextEncoder) {
    this.mentionEncoder = mentionEncoder;
    this.contextEncoder = contextEncoder;
    this.mentionIndex = new InMemoryIndex(mentionEncoder);
    this.entities = new HashMap<>();
  }
  
  public void addEntity(NamedEntity entity) {
    entities.put(entity.getId(), entity);
  }
  
  public void buildIndex() {
    // we build duplicate keys KEY.x, because an entity can have multiple names
    Map<String, String> aliases = new TreeMap<>();
    entities.values().forEach(entity -> {
      AtomicInteger idx = new AtomicInteger(0);
      entity.getAliases().forEach(a -> {
        aliases.put(entity.getId() + "." + idx.getAndIncrement(), a);
      });
    });
    mentionIndex.buildKeyIndex(aliases.keySet(), false);
    mentionIndex.encodeAndBuildVectorIndex(aliases, false);
  }
  
  public NamedEntity getEntity(String id) {
    return entities.get(id);
  }
  
  public List<EntityCandidate> findNearestEntities(String mention, int k) {
    INDArray key = mentionIndex.encode(mention);
    List<IVectorIndex.IndexEntry> entries = mentionIndex.find(key, k * 10);
    AtomicInteger idx = new AtomicInteger(0);
    return entries.stream()
      .map(entry -> {
        EntityCandidate candidate = new EntityCandidate();
        String id = entry.key.replaceFirst("\\.\\d+$", "");
        candidate.entity = getEntity(id);
        candidate.similarity = entry.similarity;
        candidate.index = idx.incrementAndGet();
        return candidate;
      })
      .distinct()
      .limit(k)
      .collect(Collectors.toList());
  }
  
  protected int countMentions() {
    return mentionIndex.size();
  }
  
  public int countEntities() {
    return entities.size();
  }
  
  class EntityCandidate implements Comparable<EntityCandidate> {
    
    public int index;
    public NamedEntity entity;
    public double similarity;
    
    @Override
    public int compareTo(@NotNull EntityCandidate o) {
      return Double.compare(similarity, o.similarity);
    }
  
    @Override
    public boolean equals(Object o) {
      if(this == o) return true;
      if(o == null || getClass() != o.getClass()) return false;
      EntityCandidate that = (EntityCandidate) o;
      return entity.equals(that.entity);
    }
  
    @Override
    public int hashCode() {
      return Objects.hash(entity);
    }
  
    public String toString() {
      return String.format(Locale.ROOT, "%s (%.2f)", entity.getId(), similarity);
    }
    
  }
  
  
}
