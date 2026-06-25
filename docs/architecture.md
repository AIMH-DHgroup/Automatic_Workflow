# Architecture

The workflow has three stages:

1. **M1 Narrative creation**: converts `.txt` or dataset `.csv` inputs into narrative CSV rows.
2. **M2 Entity enrichment**: extracts entities with Ollama and resolves Wikipedia titles to Wikidata QIDs, coordinates, and images.
3. **M3 Graph/storymap generation**: builds StoryMap HTML, Triplify JSON, and OWL output through Java Triplify.

The top-level pipeline runs the stages in order and copies each stage output into the next stage input directory.

External services used by the workflow include Ollama, Wikipedia API, Wikidata SPARQL, QLever OSM, and Java/Triplify.
