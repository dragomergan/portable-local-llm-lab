# The Winter of Lanterns: answer key (for the instructor)

Original text, written for this kit. No real-world sources, so a model cannot answer from memory: every correct answer must come from the text.

## Entities and their other names

| Canonical name | Type | Other names used in the text |
| --- | --- | --- |
| Isolde Varn | Person | Isolde, the Pale Queen, Queen Isolde |
| Cassian Varn | Person | Cassian, the Grey Warden |
| Aldous Varn | Person | King Aldous |
| Maelric Voss | Person | Voss, the Iron Chancellor, the Chancellor |
| Orren Thale | Person | Orren, the Quiet Scribe |
| Seraphine Dusk | Person | Seraphine, Mother Lantern |
| Edmund Dusk | Person | (none) |
| Tamsin Reed | Person | Tamsin |
| Brandt Halloran | Person | Admiral Halloran, Halloran |
| Dorn Hask | Person | Cassian's captain |
| Marrowfell, Ashgate, Kharn Hold, Kharn Mountains, River Sable, Ember Isles | Place | |
| House Varn, Council of Nine, Order of the Lantern, Corsair League | Group | the Order, the Council |
| Ember Crown, Sunless Key, Charter of House Varn, Glass Library | Item / Place | the Charter |
| Frost Treaty, Night of Lanterns | Event | |

Alias traps: five characters have a nickname that shares no word with their name (Pale Queen, Grey Warden, Iron Chancellor, Quiet Scribe, Mother Lantern). Automatic merging cannot catch these; the students must add them to `aliases.json`.

## Expected facts (edges)

- Isolde Varn: ruler_of Marrowfell; child_of Aldous Varn; sibling_of Cassian Varn; holds Ember Crown; spouse_of Brandt Halloran (betrothed, then married); she exiled Cassian Varn to the Ember Isles.
- Cassian Varn: warden of / guards Kharn Hold; ally_of Maelric Voss; betrayed Isolde Varn.
- Maelric Voss: leader_of Council of Nine; enemy_of Seraphine Dusk; killed Edmund Dusk (executed); opposed Frost Treaty; lost office at the end.
- Orren Thale: serves Order of the Lantern; archivist; holds Sunless Key; relative_of Tamsin Reed (her uncle); killed by Dorn Hask.
- Seraphine Dusk: leader_of Order of the Lantern; mentor_of Orren Thale; widow of Edmund Dusk; takes Orren's place as archivist.
- Tamsin Reed: carried the Charter to Brandt Halloran.
- Brandt Halloran: leader_of Corsair League; proposed Frost Treaty.
- Dorn Hask: serves Cassian Varn; killed Orren Thale.

Relation labels from the model will differ ("niece_of" vs "relative_of", "head_of" vs "leader_of"). That is expected: it is why the script has a label clean-up table.

## Test questions

| # | Question | Correct answer | What it tests |
| --- | --- | --- | --- |
| 1 | Who is the Pale Queen? | Isolde Varn | Alias |
| 2 | Who killed Orren Thale? | Dorn Hask (Cassian's captain) | One fact |
| 3 | Who was the mentor of Tamsin Reed's uncle? | Seraphine Dusk | Two steps: Tamsin, her uncle Orren, his mentor |
| 4 | Which two characters worked together against Queen Isolde? | Maelric Voss and Cassian Varn | Relation spread over several sentences |
| 5 | Who carried the Charter out of Ashgate, and to whom? | Tamsin Reed, to Admiral Halloran | Two facts joined |
| 6 | What happened to Cassian at the end? | Exiled to the Ember Isles | Late event |
| 7 | Why did Seraphine hate Voss? | He executed her husband, Edmund Dusk | Cause, far from her name in the text |
| 8 | How many characters have the surname Varn? | 3 (Aldous, Isolde, Cassian) | Counting: easy from the graph, weak for plain RAG |
| 9 | Who was Isolde's mother? | The text does not say | Honesty: a good answer is "cannot tell" |
| 10 | How old is Tamsin Reed? | The text does not say | Same |

## Suggested class exercise

1. Ask questions 1 to 10 with plain RAG. Note which fail (likely: 3, 8, and the "cannot tell" ones if the model guesses).
2. Run `graph_build.py --aliases none`. Read the edge list and write down every duplicate or wrong fact.
3. Build `aliases.json` yourself, run again, and compare.
4. Ask the same questions again with the graph facts added to the prompt.

These expectations are predictions about how a 4B model behaves; the real results depend on your model and settings, so run it once before class.
